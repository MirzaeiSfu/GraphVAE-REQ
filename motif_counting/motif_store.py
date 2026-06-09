# motif_counting/motif_store.py

import os
import torch
import pickle
from pathlib import Path
from typing import Dict, List
from decimal import Decimal
from pymysql import connect
from pymysql.err import OperationalError, MySQLError
from pandas import DataFrame
from itertools import permutations
from math import log


def get_motif_cache_dir(args=None) -> Path:
    configured_dir = getattr(args, 'motif_cache_dir', None) if args is not None else None
    if configured_dir is not None:
        return Path(configured_dir).expanduser()
    return Path(os.environ.get("MOTIF_CACHE_DIR", "cache_motifs")).expanduser()


def use_syntactic_literal_rules(args=None) -> bool:
    return getattr(args, 'use_syntactic_literal_rules', True) if args is not None else True


def get_motif_pickle_path(database_name: str, args=None) -> Path:
    marker = "injectedUC" if use_syntactic_literal_rules(args) else "noInjectedUC"
    return get_motif_cache_dir(args) / f"{database_name}_{marker}.pkl"


class RuleBasedMotifStore:
    """
    Container for motif definitions stored as relational database rules.
    Automatically handles pickle file management and database connection.

    On initialization:
    - Checks for <motif_cache_dir>/<database_name>.pkl
    - If exists: loads from pickle
    - If not: connects to database, reads data, and saves to pickle
    """

    def __init__(self, database_name: str, args, host='localhost', user='fbuser', password=''):
        """
        Initialize the motif store with automatic pickle management.

        Args:
            database_name: Name of the database (e.g., 'cora', 'citeseer')
            args: Arguments object containing configuration
            host: Database host
            user: Database user
            password: Database password
        """
        self.database_name = database_name
        self.args = args
        self.host = host
        self.user = user
        self.password = password
        self.use_syntactic_literal_rules = use_syntactic_literal_rules(args)

        # Initialize data structures
        self._initialize_structures()

        # Determine pickle path
        db_dir = get_motif_cache_dir(args)
        db_dir.mkdir(parents=True, exist_ok=True)
        self.pickle_path = get_motif_pickle_path(database_name, args)

        # Load or create motif data
        if self.pickle_path.exists():
            print(f"  📦 Found existing pickle: {self.pickle_path}")
        else:
            print(f"  🗄️  No pickle found, reading from database...")
            self._read_from_database()
            self._save_to_pickle()

    def _initialize_structures(self):
        """Initialize all data structures."""
        # Rule-related data structures
        self.rules: List = []
        self.multiples: List = []
        self.states: List = []
        self.values: List = []          # points at full or pruned depending on context

        # Both value sets stored in pickle so rule_prune can be toggled without
        # deleting the cache.
        self.values_full:   List = []   # all rows (rule_prune=False)
        self.values_pruned: List = []   # statistically significant rows (rule_prune=True)

        # Structural metadata for rules
        self.functors: Dict = {}
        self.variables: Dict = {}
        self.nodes: Dict = {}
        self.masks: Dict = {}

        # Index structures for efficient computation
        self.base_indices: List = []
        self.mask_indices: List = []
        self.sort_indices: List = []
        self.stack_indices: List = []

        # Database entities and relations
        self.entities: Dict = {}
        self.relations: Dict = {}
        self.attributes: Dict = {}
        self.keys: Dict = {}
        self.indices: Dict = {}
        self.matrices: Dict = {}

        # Feature mapping structures
        self.entity_feature_columns: Dict = {}
        self.entity_literal_values: Dict = {}
        self.relation_feature_columns: Dict = {}
        self.relation_entity_tables: Dict = {}
        self.relation_literal_values: Dict = {}
        self.relation_occurrence_counts: Dict = {}
        self.feature_info_mapping: Dict = {}

        # Configuration
        self.device = getattr(self.args, 'device', 'cuda')
        self.num_nodes_graph: int = 0
        self.total_relation_occurrences: Dict = {}

    @property
    def num_motifs(self) -> int:
        """Total number of motif rules."""
        return len(self.rules)


    def _save_to_pickle(self):
        """Save all data to pickle file."""
        print(f"  💾 Saving to pickle: {self.pickle_path}")

        # Convert tensors to CPU for saving
        matrices_cpu = {}
        for key, matrix in self.matrices.items():
            if isinstance(matrix, torch.Tensor):
                matrices_cpu[key] = matrix.cpu()
            else:
                matrices_cpu[key] = matrix

        # Prepare data dictionary
        data = {
            "entities": self.entities,
            "relations": self.relations,
            "keys": self.keys,
            "matrices": matrices_cpu,
            "rules": self.rules,
            "indices": self.indices,
            "attributes": self.attributes,
            "base_indices": self.base_indices,
            "mask_indices": self.mask_indices,
            "sort_indices": self.sort_indices,
            "stack_indices": self.stack_indices,
            # Both value sets — motif_counter selects at load time based on --rule_prune
            "values_full":   self.values_full,
            "values_pruned": self.values_pruned,
            "functors": self.functors,
            "variables": self.variables,
            "nodes": self.nodes,
            "states": self.states,
            "masks": self.masks,
            "multiples": self.multiples,
            "entity_feature_columns": self.entity_feature_columns,
            "entity_literal_values": self.entity_literal_values,
            "relation_feature_columns": self.relation_feature_columns,
            "relation_entity_tables": self.relation_entity_tables,
            "relation_literal_values": self.relation_literal_values,
            "relation_occurrence_counts": self.relation_occurrence_counts,
            "feature_info_mapping": self.feature_info_mapping,
            "num_nodes_graph": self.num_nodes_graph,
            "total_relation_occurrences": self.total_relation_occurrences,
            "use_syntactic_literal_rules": self.use_syntactic_literal_rules,
        }

        with open(self.pickle_path, "wb") as f:
            pickle.dump(data, f)

        print(f"  ✓ Saved to {self.pickle_path}")

    def _read_from_database(self):
        """Read all data from MySQL database."""
        try:
            connections = self._connect_to_databases()

            try:
                print("    • Reading entities...")
                self._fetch_entities(connections['main'], connections['setup'])

                print("    • Reading relations...")
                self._fetch_relations(connections['main'], connections['setup'])

                print("    • Reading attributes...")
                self._fetch_attributes(connections['setup'])

                print("    • Creating indices...")
                self._create_indices()

                print("    • Creating mask matrices...")
                self._create_mask_matrices(connections['setup'])

                print("    • Processing Bayesian Network rules...")
                self._process_rules(connections['bn'], connections['setup'])

                print("    • Creating feature mappings...")
                self._create_feature_info_mapping()

                print(f"  ✓ Successfully read all data from database")

            finally:
                self._close_connections(connections)

        except (OperationalError, MySQLError) as e:
            error_msg = (
                f"\n✗ Database connection failed: {e}\n"
                f"  Please ensure:\n"
                f"    1. MySQL is running\n"
                f"    2. Database '{self.database_name}' exists\n"
                f"    3. Database credentials are correct"
            )
            raise RuntimeError(error_msg)

    def _connect_to_databases(self) -> Dict:
        """Establish connections to main, setup, and Bayesian Network databases."""
        connections = {}

        try:
            # Main database
            conn_main = connect(host=self.host, user=self.user, password=self.password, db=self.database_name)
            connections['main'] = {'connection': conn_main, 'cursor': conn_main.cursor()}

            # Setup database
            db_setup = f"{self.database_name}_setup"
            conn_setup = connect(host=self.host, user=self.user, password=self.password, db=db_setup)
            connections['setup'] = {'connection': conn_setup, 'cursor': conn_setup.cursor()}

            # Bayesian Network database
            db_bn = f"{self.database_name}_BN"
            conn_bn = connect(host=self.host, user=self.user, password=self.password, db=db_bn)
            connections['bn'] = {'connection': conn_bn, 'cursor': conn_bn.cursor()}

        except OperationalError as e:
            for conn_dict in connections.values():
                try:
                    conn_dict['cursor'].close()
                    conn_dict['connection'].close()
                except:
                    pass
            raise

        return connections

    def _fetch_entities(self, main_conn, setup_conn):
        """Fetch entity tables and their primary keys."""
        cursor_main = main_conn['cursor']
        cursor_setup = setup_conn['cursor']

        cursor_setup.execute("SELECT TABLE_NAME FROM EntityTables")
        entity_tables = cursor_setup.fetchall()

        for (table_name,) in entity_tables:
            cursor_main.execute(f"SELECT * FROM {table_name}")
            rows = cursor_main.fetchall()

            cursor_main.execute(f"SHOW COLUMNS FROM {self.database_name}.{table_name}")
            columns = cursor_main.fetchall()
            columns_names = [column[0] for column in columns]

            self.entities[table_name] = DataFrame(rows, columns=columns_names)
            self.entity_feature_columns[table_name] = columns_names[1:]
            if self.use_syntactic_literal_rules:
                self.entity_literal_values[table_name] = {
                    feature_name: self._get_unique_literal_values(self.entities[table_name][feature_name])
                    for feature_name in self.entity_feature_columns[table_name]
                }

            cursor_setup.execute("SELECT COLUMN_NAME FROM EntityTables WHERE TABLE_NAME = %s", (table_name,))
            key = cursor_setup.fetchall()
            self.keys[table_name] = key[0][0]

    def _fetch_relations(self, main_conn, setup_conn):
        """Fetch relation tables and their foreign keys."""
        cursor_main = main_conn['cursor']
        cursor_setup = setup_conn['cursor']

        cursor_setup.execute("SELECT TABLE_NAME FROM RelationTables")
        relation_tables = cursor_setup.fetchall()

        for (table_name,) in relation_tables:
            cursor_main.execute(f"SELECT * FROM {table_name}")
            rows = cursor_main.fetchall()

            cursor_main.execute(f"SHOW COLUMNS FROM {self.database_name}.{table_name}")
            columns = cursor_main.fetchall()
            columns_names = [column[0] for column in columns]

            self.relations[table_name] = DataFrame(rows, columns=columns_names)
            self.relation_feature_columns[table_name] = columns_names[2:]
            if self.use_syntactic_literal_rules:
                self.relation_literal_values[table_name] = {
                    feature_name: self._get_unique_literal_values(self.relations[table_name][feature_name])
                    for feature_name in self.relation_feature_columns[table_name]
                }
                self.relation_occurrence_counts[table_name] = len(self.relations[table_name])
                self.total_relation_occurrences[table_name] = self.relation_occurrence_counts[table_name]

            cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
            key = cursor_setup.fetchall()
            self.keys[table_name] = (key[0][0], key[1][0])

            if self.use_syntactic_literal_rules:
                cursor_setup.execute("SELECT REFERENCED_TABLE_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
                references = cursor_setup.fetchall()
                self.relation_entity_tables[table_name] = (references[0][0], references[1][0])

    def _fetch_attributes(self, setup_conn):
        """Fetch attribute columns."""
        cursor_setup = setup_conn['cursor']
        cursor_setup.execute("SELECT COLUMN_NAME, TABLE_NAME FROM AttributeColumns")
        attribute_columns = cursor_setup.fetchall()

        for column_name, table_name in attribute_columns:
            self.attributes[column_name] = table_name

    def _create_indices(self):
        """Create indices for quick lookup of entity keys."""
        for table_name, df in self.entities.items():
            key = self.keys[table_name]
            self.indices[key] = {row[key]: idx for idx, row in df.iterrows()}

    @staticmethod
    def _get_unique_literal_values(series):
        """Return `(value, count)` tuples for the literals in a feature column."""
        value_counts = series.dropna().value_counts(sort=False)
        value_count_pairs = []
        for value, count in value_counts.items():
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except ValueError:
                    pass
            if hasattr(count, "item"):
                try:
                    count = count.item()
                except ValueError:
                    pass
            value_count_pairs.append((value, count))

        try:
            value_count_pairs = sorted(value_count_pairs, key=lambda pair: pair[0])
        except TypeError:
            pass
        return value_count_pairs

    def _create_mask_matrices(self, setup_conn):
        """Create mask matrices representing relations between entities."""
        cursor_setup = setup_conn['cursor']

        # Initialize matrices
        for table_name, df in self.relations.items():
            cursor_setup.execute("SELECT REFERENCED_TABLE_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
            reference = cursor_setup.fetchall()
            entity1 = reference[0][0]
            entity2 = reference[1][0]

            shape = (10, 10)
            self.matrices[table_name] = torch.zeros(shape, dtype=torch.float32, device=self.device)

        # Populate matrices
        # for table_name, df in self.relations.items():
        #     cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
        #     key = cursor_setup.fetchall()
        #     cursor_setup.execute("SELECT COLUMN_NAME, REFERENCED_COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
        #     reference = cursor_setup.fetchall()

        #     rows_indices = []
        #     cols_indices = []
        #     for index, row in df.iterrows():
        #         row_index = self.indices[reference[0][1]][row[key[0][0]]]
        #         col_index = self.indices[reference[1][1]][row[key[1][0]]]
        #         rows_indices.append(row_index)
        #         cols_indices.append(col_index)

        #     rows_indices_tensor = torch.tensor(rows_indices, dtype=torch.long)
        #     cols_indices_tensor = torch.tensor(cols_indices, dtype=torch.long)
        #     self.matrices[table_name][rows_indices_tensor, cols_indices_tensor] = 1

    def _process_rules(self, bn_conn, setup_conn):
        """Process rules from Bayesian Network and prepare for counting."""
        cursor_bn = bn_conn['cursor']
        cursor_setup = setup_conn['cursor']

        cursor_bn.execute("SELECT DISTINCT child FROM Final_Path_BayesNets_view")
        childs = cursor_bn.fetchall()

        relation_names = tuple(self.relations.keys())

        for i in range(len(childs)):
            rule = [childs[i][0]]
            cursor_bn.execute("SELECT parent FROM Final_Path_BayesNets_view WHERE child = %s", (childs[i][0],))
            parents = cursor_bn.fetchall()
            for (parent,) in parents:
                if parent != '':
                    rule.append(parent)

            cursor_bn.execute(f"SELECT * FROM `{childs[i][0]}_CP`")
            value = cursor_bn.fetchall()
            self._add_processed_rule(rule, value, relation_names)

        if self.use_syntactic_literal_rules:
            self._ensure_entity_unary_literal_rules(relation_names)
            self._ensure_relation_literal_rules(relation_names)

        self._adjust_matrices()

    def _add_processed_rule(self, rule, value_rows, relation_names, keep_all_values=False):
        """Add one rule and populate all aligned rule metadata structures."""
        rule_idx = len(self.rules)
        self.rules.append(rule)
        self.multiples.append(1 if len(rule) > 1 else 0)

        relation_check = any(',' in atom for atom in rule)
        functor, variable, node, state, mask = {}, {}, {}, [], {}
        unmasked_variables = []

        for j in range(len(rule)):
            fun = rule[j].split('(')[0]
            functor[j] = fun

            if ',' not in rule[j]:
                var = rule[j].split('(')[1][:-1]
                variable[j] = var
                node[j] = var[:-1]

                if not relation_check:
                    unmasked_variables.append(var)
                    state.append(0)
                else:
                    mas = []
                    for k in rule:
                        func = k.split('(')[0]
                        if func not in relation_names:
                            func = self.attributes.get(func, func)
                        if ',' in k and var in k:
                            var1, var2 = k.split('(')[1][:-1].split(',')
                            mas.append([func, var1, var2])
                            unmasked_variables.append(k.split('(')[1][:-1])
                    mask[j] = mas
                    state.append(1)
            else:
                unmasked_variables.append(rule[j].split('(')[1][:-1])
                if fun in relation_names:
                    state.append(2)
                else:
                    state.append(3)

        self.functors[rule_idx] = functor
        self.variables[rule_idx] = variable
        self.nodes[rule_idx] = node
        self.states.append(state)
        self.masks[rule_idx] = mask

        masked_variables = [unmasked_variables[0]]
        base_indice = [0]
        mask_indice = []

        for j in range(1, len(unmasked_variables)):
            mask_check = False
            for k in range(len(masked_variables)):
                if unmasked_variables[j] == masked_variables[k]:
                    mask_indice.append([k, j])
                    mask_check = True
                    break
            if not mask_check:
                base_indice.append(j)
                masked_variables.append(unmasked_variables[j])

        sort_indice, sorted_variables = self._create_sort_indices(masked_variables, relation_check, relation_names)
        stack_indice = self._create_stack_indices(sorted_variables)

        self.base_indices.append(base_indice)
        self.mask_indices.append(mask_indice)
        self.sort_indices.append(sort_indice)
        self.stack_indices.append(stack_indice)

        # Remove N/A rows regardless of pruning setting.
        value_rows = [row for row in value_rows if 'N/A' not in row]

        # ── Always compute BOTH value sets so a single pickle works
        # for either value of --rule_prune without deleting the cache.
        self.values_full.append(value_rows)

        # Unary rules are never pruned; they keep all rows.
        pruned_value = []
        if keep_all_values or len(rule) == 1:
            pruned_value = list(value_rows)
        else:
            for row in value_rows:
                size = len(row)
                try:
                    if self.multiples[rule_idx]:
                        if 2 * row[size-4] * (log(row[size-3]) - log(row[size-1])) - log(row[size-4]) > 0:
                            pruned_value.append(row)
                    else:
                        if 2 * int(row[size-3]) * (log(row[size-5]) - log(row[size-1])) - log(int(row[size-3])) > 0:
                            pruned_value.append(row)
                except (ValueError, ZeroDivisionError):
                    # log(0) or log(negative) — row has zero count/probability, skip it
                    pass
        self.values_pruned.append(pruned_value)

        # Keep self.values pointing at full for any in-memory use within
        # motif_store (e.g. _adjust_matrices). motif_counter re-selects at load.
        self.values.append(value_rows)

    def _ensure_entity_unary_literal_rules(self, relation_names):
        """Add missing unary rules for entity literals using entity feature values."""
        unary_functors = {
            rule[0].split('(')[0]
            for rule in self.rules
            if len(rule) == 1
        }

        for table_name, feature_values in self.entity_literal_values.items():
            variable_name = f"{table_name}0"
            for feature_name, value_rows in feature_values.items():
                if feature_name in unary_functors:
                    continue
                unary_rule = [f"{feature_name}({variable_name})"]
                synthetic_value_rows = self._build_synthetic_unary_value_rows(value_rows)
                self._add_processed_rule(
                    unary_rule,
                    synthetic_value_rows,
                    relation_names,
                    keep_all_values=True,
                )
                unary_functors.add(feature_name)

    def _ensure_relation_literal_rules(self, relation_names):
        """Add missing binary edge-feature and standalone relation rules."""
        existing_rules = {tuple(rule) for rule in self.rules}

        for relation_name, feature_values in self.relation_literal_values.items():
            entity1, entity2 = self.relation_entity_tables[relation_name]
            variable1 = f"{entity1}0"
            variable2 = f"{entity2}1"
            relation_atom = f"{relation_name}({variable1},{variable2})"

            for feature_name in feature_values:
                feature_rule = (
                    f"{feature_name}({variable1},{variable2})",
                    relation_atom,
                )
                if feature_rule not in existing_rules:
                    synthetic_value_rows = self._build_synthetic_relation_feature_value_rows(
                        feature_values[feature_name]
                    )
                    self._add_processed_rule(
                        list(feature_rule),
                        synthetic_value_rows,
                        relation_names,
                        keep_all_values=True,
                    )
                    existing_rules.add(feature_rule)

            standalone_rule = (relation_atom,)
            if standalone_rule not in existing_rules:
                synthetic_value_rows = self._build_synthetic_standalone_relation_value_rows(
                    self.relation_occurrence_counts[relation_name]
                )
                self._add_processed_rule(
                    [relation_atom],
                    synthetic_value_rows,
                    relation_names,
                    keep_all_values=True,
                )
                existing_rules.add(standalone_rule)

    @staticmethod
    def _build_synthetic_unary_value_rows(value_rows):
        """Convert `(value, count)` pairs into 5-element rows for synthetic unary rules."""
        synthetic_rows = []
        for value, count in value_rows:
            synthetic_rows.append([
                str(value),
                Decimal(str(count)),
                Decimal(str(count)),
                "",
                "",
            ])
        return synthetic_rows

    @staticmethod
    def _build_synthetic_relation_feature_value_rows(value_rows):
        """Convert `(value, count)` pairs into synthetic binary rule rows."""
        synthetic_rows = []
        for value, count in value_rows:
            count_decimal = Decimal(str(count))
            synthetic_rows.append([
                count_decimal,
                value,
                "T",
                "",
                count_decimal,
                "",
                "",
                "",
            ])
        return synthetic_rows

    @staticmethod
    def _build_synthetic_standalone_relation_value_rows(count):
        """Build the single synthetic row for a standalone relation rule."""
        count_decimal = Decimal(str(count))
        return [[
            "T",
            1.0,
            count_decimal,
            count_decimal,
            0.0,
            1.0,
        ]]

    def _create_sort_indices(self, masked_variables, relation_check, relation_names):
        """Create indices to sort variables for matrix multiplication chain."""
        sort_indice = []
        sorted_variables = []

        if not relation_check:
            sort_indice.append([False, 0])
            sorted_variables.append(masked_variables[0])
        else:
            indices_permutations = list(permutations(range(len(masked_variables))))
            variables_permutations = list(permutations(masked_variables))
            found_chain = False

            for idx_perm, var_perm in zip(indices_permutations, variables_permutations):
                indices_chain = []
                variables_chain = []
                first = var_perm[0].split(',')[0]
                second = var_perm[0].split(',')[1]
                indices_chain.append([False, idx_perm[0]])
                variables_chain.append(var_perm[0])
                untransposed_check = True

                for k in range(1, len(var_perm)):
                    next_first = var_perm[k].split(',')[0]
                    next_second = var_perm[k].split(',')[1]
                    if second == next_first:
                        second = next_second
                        indices_chain.append([False, idx_perm[k]])
                        variables_chain.append(var_perm[k])
                    elif second == next_second:
                        second = next_first
                        indices_chain.append([True, idx_perm[k]])
                        variables_chain.append(next_second + ',' + next_first)
                    else:
                        untransposed_check = False
                        break

                if untransposed_check:
                    sort_indice = indices_chain
                    sorted_variables = variables_chain
                    found_chain = True
                    break

        return sort_indice, sorted_variables

    def _create_stack_indices(self, sorted_variables):
        """Create indices for stacking matrices in correct order."""
        stack_indices = []
        for j in range(1, len(sorted_variables)):
            second = sorted_variables[j].split(',')[1]
            for k in range(j - 1, -1, -1):
                previous_first = sorted_variables[k].split(',')[0]
                if previous_first == second:
                    stack_indices.append([k, j])
        return stack_indices

    def _adjust_matrices(self):
        """Adjust matrices to correct shape by transposing if necessary."""
        relation_functors = [item for sublist in self.rules for item in sublist
                           if ',' in item and item in self.relations.keys()]
        unique_relation_functors = list(set(relation_functors))

        for relation_functor in unique_relation_functors:
            entities_involved = relation_functor.replace(')', '').split('(')[1].split(',')
            entities_clean = [entity[:-1] for entity in entities_involved]
            correct_shape = (len(self.entities[entities_clean[0]]), len(self.entities[entities_clean[1]]))
            matrix_name = relation_functor.split('(')[0]

            if self.matrices[matrix_name].shape != correct_shape:
                self.matrices[matrix_name] = self.matrices[matrix_name].t()

    def _create_feature_info_mapping(self):
        """Create feature info mapping for all edge features in all relations."""
        num_nodes = 0
        for relation_name, relation_df in self.relations.items():
            all_columns = list(relation_df.columns)
            node_id_cols = all_columns[:2]
            max_node = max(relation_df[node_id_cols[0]].max(), relation_df[node_id_cols[1]].max())
            num_nodes = max(num_nodes, max_node + 1)

        self.num_nodes_graph = num_nodes

        feature_index = 0
        for relation_name, relation_df in self.relations.items():
            all_columns = list(relation_df.columns)
            node_id_cols = all_columns[:2]
            feature_columns = all_columns[2:]

            for feature_col in feature_columns:
                unique_values = sorted(relation_df[feature_col].unique())
                value_index_mapping = {i: int(val) for i, val in enumerate(unique_values)}
                num_unique_values = len(unique_values)
                tensor_shape = [num_unique_values, num_nodes, num_nodes]

                self.feature_info_mapping[feature_index] = {
                    'relation_name': relation_name,
                    'feature_name': feature_col,
                    'value_index_mapping': value_index_mapping,
                    'node_id_columns': node_id_cols,
                    'tensor_shape': tensor_shape
                }
                feature_index += 1

    def _close_connections(self, connections: Dict):
        """Close all database connections."""
        for conn_dict in connections.values():
            try:
                conn_dict['cursor'].close()
                conn_dict['connection'].close()
            except:
                pass

    def __repr__(self):
        return f"RuleBasedMotifStore(database={self.database_name}, num_motifs={self.num_motifs}, num_entities={len(self.entities)}, num_relations={len(self.relations)})"
