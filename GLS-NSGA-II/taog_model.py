import itertools
from typing import List, Dict, Tuple


class Node:
    def __init__(self, node_id: str, node_type: str):
        """
        node_type: 'A' (Subassembly/Artificial), 'B' (Task/Normal)
        """
        self.id = node_id
        self.type = node_type
        self.children = []


class Alternative:
    """表示从TAOG中提取出的一种可行的拆卸方案（包含任务集合和优先级边）"""

    def __init__(self):
        self.tasks = []  # 包含的任务ID列表 (B节点)
        self.edges = []  # 优先级边列表 (pred, succ)
        self.entry_tasks = []  # 该方案的入口任务（无前驱的任务）


class TAOG:
    def __init__(self):
        self.nodes = {}
        self.root_id = None

    def add_node(self, node_id: str, node_type: str):
        self.nodes[node_id] = Node(node_id, node_type)

    def add_edge(self, parent_id: str, child_id: str):
        self.nodes[parent_id].children.append(self.nodes[child_id])

    def set_root(self, root_id: str):
        self.root_id = root_id

    def extract_all_alternatives(self) -> List[Alternative]:


        def dfs(node: Node) -> List[Alternative]:
            if node.type == 'A':
                # OR 关系：子装配体可以通过多条路径拆卸，分支延伸
                alts = []
                for child in node.children:
                    alts.extend(dfs(child))
                return alts

            elif node.type == 'B':
                # AND 关系：任务拆卸后产生多个子装配体，必须全部拆卸（笛卡尔积组合）
                child_alts_list = [dfs(child) for child in node.children]

                if not child_alts_list:
                    # 叶子任务
                    alt = Alternative()
                    alt.tasks = [node.id]
                    alt.entry_tasks = [node.id]
                    return [alt]

                # 笛卡尔积组合所有后继分支的方案
                res = []
                for combo in itertools.product(*child_alts_list):
                    alt = Alternative()
                    alt.tasks = [node.id]
                    alt.entry_tasks = [node.id]

                    for sub_alt in combo:
                        alt.tasks.extend(sub_alt.tasks)
                        alt.edges.extend(sub_alt.edges)
                        # 当前任务是其所有直接后继分支入口任务的前驱
                        for entry in sub_alt.entry_tasks:
                            alt.edges.append((node.id, entry))
                    res.append(alt)
                return res

        return dfs(self.nodes[self.root_id])