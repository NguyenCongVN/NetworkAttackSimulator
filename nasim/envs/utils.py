import enum
import numpy as np
from queue import deque
from itertools import permutations

INTERNET = 0


class OneHotBool(enum.IntEnum):
    NONE = 0
    TRUE = 1
    FALSE = 2

    @staticmethod
    def from_bool(b):
        if b:
            return OneHotBool.TRUE
        return OneHotBool.FALSE

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class ServiceState(enum.IntEnum):
    # values for possible service knowledge states
    UNKNOWN = 0     # service may or may not be running on host
    PRESENT = 1     # service is running on the host
    ABSENT = 2      # service not running on the host

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class AccessLevel(enum.IntEnum):
    NONE = 0
    USER = 1
    ROOT = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def get_minimal_hops_to_goal(topology, sensitive_addresses):
    """Tính số bước nhảy tối thiểu trong mạng để đến tất cả các host nhạy cảm.
    
    Bắt đầu từ bên ngoài mạng (tức là chỉ có thể tiếp cận các subnet được phơi ra).
    
    Thuật toán hoạt động bằng cách:
    1. Tính toán khoảng cách ngắn nhất giữa tất cả các cặp subnet sử dụng thuật toán Floyd-Warshall
    2. Tìm đường đi ngắn nhất từ Internet đến tất cả các subnet chứa host nhạy cảm bằng cách xem xét
       tất cả các hoán vị có thể
    
    Parameters
    ----------
    topology : list hoặc numpy.ndarray
        Ma trận kề biểu diễn cấu trúc liên kết mạng, trong đó topology[i][j] = 1 
        nếu subnet i kết nối với subnet j, ngược lại là 0
    sensitive_addresses : list
        Danh sách các địa chỉ host nhạy cảm, mỗi địa chỉ là một tuple (subnet_id, host_id)
    
        Số bước nhảy mạng tối thiểu để đến tất cả các host nhạy cảm
    
    Notes
    -----
    - Độ phức tạp thời gian: O(n³ + m!), với n là số lượng subnet và m là số lượng subnet chứa host nhạy cảm
    - Phần Floyd-Warshall có độ phức tạp O(n³)
    - Phần tính toán tất cả các hoán vị có độ phức tạp O(m!)
    """
    num_subnets = len(topology)
    max_value = np.iinfo(np.int16).max
    distance = np.full((num_subnets, num_subnets),
                       max_value,
                       dtype=np.int16)

    # set distances for each edge to 1
    for s1 in range(num_subnets):
        for s2 in range(num_subnets):
            if s1 == s2:
                distance[s1][s2] = 0
            elif topology[s1][s2] == 1:
                distance[s1][s2] = 1
    # find all pair minimum shortest path distance
    for k in range(num_subnets):
        for i in range(num_subnets):
            for j in range(num_subnets):
                if distance[i][k] == max_value \
                   or distance[k][j] == max_value:
                    dis = max_value
                else:
                    dis = distance[i][k] + distance[k][j]
                if distance[i][j] > dis:
                    distance[i][j] = distance[i][k] + distance[k][j]

    # get list of all subnets we need to visit
    subnets_to_visit = [INTERNET]
    for subnet, host in sensitive_addresses:
        if subnet not in subnets_to_visit:
            subnets_to_visit.append(subnet)

    # find minimum shortest path that visits internet subnet and all
    # sensitive subnets by checking all possible permutations
    shortest = max_value
    for pm in permutations(subnets_to_visit):
        pm_sum = 0
        for i in range(len(pm) - 1):
            pm_sum += distance[pm[i]][pm[i+1]]
        shortest = min(shortest, pm_sum)

    return shortest


def min_subnet_depth(topology):
    """Tìm độ sâu tối thiểu của mỗi subnet trong đồ thị mạng, tính bằng số bước
    từ subnet tiếp xúc với internet đến mỗi subnet

    Tham số
    topology : ma trận 2D
        Ma trận kề đại diện cho mạng, với subnet đầu tiên
        đại diện cho internet (tức là tiếp xúc với bên ngoài)

    Giá trị trả về
        Danh sách độ sâu của mỗi subnet, được sắp xếp theo chỉ số subnet trong topology
    """
    num_subnets = len(topology)

    assert len(topology[0]) == num_subnets

    depths = []
    Q = deque()
    for subnet in range(num_subnets):
        if topology[subnet][INTERNET] == 1:
            depths.append(0)
            Q.appendleft(subnet)
        else:
            depths.append(float('inf'))

    while len(Q) > 0:
        parent = Q.pop()
        for child in range(num_subnets):
            if topology[parent][child] == 1:
                # child is connected to parent
                if depths[child] > depths[parent] + 1:
                    depths[child] = depths[parent] + 1
                    Q.appendleft(child)
    return depths
