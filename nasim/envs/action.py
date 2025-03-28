"""Action related classes for the NASim environment.

This module contains the different action classes that are used
to implement actions within a NASim environment, along within the
different ActionSpace classes, and the ActionResult class.

Notes
-----

**Actions:**

Every action inherits from the base :class:`Action` class, which defines
some common attributes and functions. Different types of actions
are implemented as subclasses of the Action class.

Action types implemented:

- :class:`Exploit`
- :class:`PrivilegeEscalation`
- :class:`ServiceScan`
- :class:`OSScan`
- :class:`SubnetScan`
- :class:`ProcessScan`
- :class:`NoOp`

**Action Spaces:**

There are two types of action spaces, depending on if you are using flat
actions or not:

- :class:`FlatActionSpace`
- :class:`ParameterisedActionSpace`

"""

import math
import numpy as np
from gymnasium import spaces

from nasim.envs.utils import AccessLevel


def load_action_list(scenario):
    """Tải danh sách các hành động có thể thực hiện trong môi trường dựa trên kịch bản

    Parameters
    ----------
    scenario : Scenario
        Đối tượng kịch bản chứa thông tin về mạng và các hành động có thể

    Returns
    -------
    list
        Danh sách tất cả các hành động có thể thực hiện trong môi trường
    """
    # Khởi tạo danh sách rỗng để chứa tất cả các hành động
    action_list = []
    
    # Duyệt qua từng địa chỉ máy chủ trong không gian địa chỉ của kịch bản
    # Mỗi địa chỉ có dạng (subnet_id, host_id)
    for address in scenario.address_space:
        # Thêm hành động quét dịch vụ (ServiceScan) cho máy chủ này
        # Hành động này giúp phát hiện các dịch vụ đang chạy trên máy chủ mục tiêu
        action_list.append(
            ServiceScan(address, scenario.service_scan_cost)
        )
        
        # Thêm hành động quét hệ điều hành (OSScan) cho máy chủ này
        # Giúp xác định hệ điều hành đang chạy trên máy chủ mục tiêu
        action_list.append(
            OSScan(address, scenario.os_scan_cost)
        )
        
        # Thêm hành động quét subnet (SubnetScan) cho máy chủ này
        # Giúp phát hiện các máy chủ khác trong cùng subnet với máy chủ này
        # Đặc biệt quan trọng để mở rộng khả năng quan sát trong chế độ fully_obs=False
        action_list.append(
            SubnetScan(address, scenario.subnet_scan_cost)
        )
        
        # Thêm hành động quét tiến trình (ProcessScan) cho máy chủ này
        # Giúp phát hiện các tiến trình đang chạy trên máy chủ, cần thiết cho việc leo thang đặc quyền
        action_list.append(
            ProcessScan(address, scenario.process_scan_cost)
        )
        
        # Duyệt qua tất cả các khai thác lỗ hổng (exploits) được định nghĩa trong kịch bản
        # e_name: tên của lỗ hổng, e_def: định nghĩa chi tiết về lỗ hổng (dịch vụ mục tiêu, OS, chi phí...)
        for e_name, e_def in scenario.exploits.items():
            # Tạo một đối tượng Exploit với tên, địa chỉ mục tiêu và các tham số từ định nghĩa
            # Các lỗ hổng này được sử dụng để xâm nhập vào máy chủ mục tiêu
            exploit = Exploit(e_name, address, **e_def)
            action_list.append(exploit)
        
        # Duyệt qua tất cả các kỹ thuật leo thang đặc quyền (privilege escalations) trong kịch bản
        # pe_name: tên kỹ thuật, pe_def: định nghĩa chi tiết (tiến trình mục tiêu, OS, chi phí...)
        for pe_name, pe_def in scenario.privescs.items():
            # Tạo đối tượng PrivilegeEscalation với tên, địa chỉ mục tiêu và các tham số
            # Kỹ thuật leo thang đặc quyền được sử dụng để nâng cao quyền truy cập trên máy chủ đã xâm nhập
            privesc = PrivilegeEscalation(pe_name, address, **pe_def)
            action_list.append(privesc)
    
    # Trả về danh sách đầy đủ các hành động có thể thực hiện trong môi trường
    return action_list


class Action:
    """The base abstract action class in the environment

    There are multiple types of actions (e.g. exploit, scan, etc.), but every
    action has some common attributes.

    ...

    Attributes
    ----------
    name : str
        the name of action
    target : (int, int)
        the (subnet, host) address of target of the action. The target of the
        action could be the address of a host that the action is being used
        against (e.g. for exploits or targeted scans) or could be the host that
        the action is being executed on (e.g. for subnet scans).
    cost : float
        the cost of performing the action
    prob : float
        the success probability of the action. This is the probability that
        the action works given that it's preconditions are met. E.g. a remote
        exploit targeting a host that you cannot communicate with will always
        fail. For deterministic actions this will be 1.0.
    req_access : AccessLevel,
        the required access level to perform action. For for on host actions
        (i.e. subnet scan, process scan, and privilege escalation) this will
        be the access on the target. For remote actions (i.e. service scan,
        os scan, and exploits) this will be the access on a pivot host (i.e.
        a compromised host that can reach the target).
    """

    def __init__(self,
                 name,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        name : str
            name of action
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        assert 0 <= prob <= 1.0
        self.name = name
        self.target = target
        self.cost = cost
        self.prob = prob
        self.req_access = req_access

    def is_exploit(self):
        """Check if action is an exploit

        Returns
        -------
        bool
            True if action is exploit, otherwise False
        """
        return isinstance(self, Exploit)

    def is_privilege_escalation(self):
        """Check if action is privilege escalation action

        Returns
        -------
        bool
            True if action is privilege escalation action, otherwise False
        """
        return isinstance(self, PrivilegeEscalation)

    def is_scan(self):
        """Check if action is a scan

        Returns
        -------
        bool
            True if action is scan, otherwise False
        """
        return isinstance(self, (ServiceScan, OSScan, SubnetScan, ProcessScan))

    def is_remote(self):
        """Check if action is a remote action

        A remote action is one where the target host is a remote host (i.e. the
        action is not performed locally on the target)

        Returns
        -------
        bool
            True if action is remote, otherwise False
        """
        return isinstance(self, (ServiceScan, OSScan, Exploit))

    def is_service_scan(self):
        """Check if action is a service scan

        Returns
        -------
        bool
            True if action is service scan, otherwise False
        """
        return isinstance(self, ServiceScan)

    def is_os_scan(self):
        """Check if action is an OS scan

        Returns
        -------
        bool
            True if action is an OS scan, otherwise False
        """
        return isinstance(self, OSScan)

    def is_subnet_scan(self):
        """Check if action is a subnet scan

        Returns
        -------
        bool
            True if action is a subnet scan, otherwise False
        """
        return isinstance(self, SubnetScan)

    def is_process_scan(self):
        """Check if action is a process scan

        Returns
        -------
        bool
            True if action is a process scan, otherwise False
        """
        return isinstance(self, ProcessScan)

    def is_noop(self):
        """Check if action is a do nothing action.

        Returns
        -------
        bool
            True if action is a noop action, otherwise False
        """
        return isinstance(self, NoOp)

    def __str__(self):
        return (f"{self.__class__.__name__}: "
                f"target={self.target}, "
                f"cost={self.cost:.2f}, "
                f"prob={self.prob:.2f}, "
                f"req_access={self.req_access}")

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        if self.target != other.target:
            return False
        if not (math.isclose(self.cost, other.cost)
                and math.isclose(self.prob, other.prob)):
            return False
        return self.req_access == other.req_access


class Exploit(Action):
    """An Exploit action in the environment

    Inherits from the base Action Class.

    ...

    Attributes
    ----------
    service : str
        the service targeted by exploit
    os : str
        the OS targeted by exploit. If None then exploit works for all OSs.
    access : int
        the access level gained on target if exploit succeeds.
    """

    def __init__(self,
                 name,
                 target,
                 cost,
                 service,
                 os=None,
                 access=0,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        service : str
            the target service
        os : str, optional
            the target OS of exploit, if None then exploit works for all OS
            (default=None)
        access : int, optional
            the access level gained on target if exploit succeeds (default=0)
        prob : float, optional
            probability of success (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)
        self.os = os
        self.service = service
        self.access = access

    def __str__(self):
        return (f"{super().__str__()}, os={self.os}, "
                f"service={self.service}, access={self.access}")

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.service == other.service \
            and self.os == other.os \
            and self.access == other.access


class PrivilegeEscalation(Action):
    """A privilege escalation action in the environment

    Inherits from the base Action Class.

    ...

    Attributes
    ----------
    process : str
        the process targeted by the privilege escalation. If None the action
        works independent of a process
    os : str
        the OS targeted by privilege escalation. If None then action works
        for all OSs.
    access : int
        the access level resulting from privilege escalation action
    """

    def __init__(self,
                 name,
                 target,
                 cost,
                 access,
                 process=None,
                 os=None,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        access : int
            the access level resulting from the privilege escalation
        process : str, optional
            the target process, if None the action does not require a process
            to work (default=None)
        os : str, optional
            the target OS of privilege escalation action, if None then action
            works for all OS (default=None)
        prob : float, optional
            probability of success (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)
        self.access = access
        self.os = os
        self.process = process

    def __str__(self):
        return (f"{super().__str__()}, os={self.os}, "
                f"process={self.process}, access={self.access}")

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.process == other.process \
            and self.os == other.os \
            and self.access == other.access


class ServiceScan(Action):
    """A Service Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("service_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class OSScan(Action):
    """An OS Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("os_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class SubnetScan(Action):
    """A Subnet Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("subnet_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class ProcessScan(Action):
    """A Process Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("process_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class NoOp(Action):
    """A do nothing action in the environment

    Inherits from the base Action Class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name="noop",
                         target=(1, 0),
                         cost=0,
                         prob=1.0,
                         req_access=AccessLevel.NONE)


class ActionResult:
    """A dataclass for storing the results of an Action.

    These results are then used to update the full state and observation.

    ...

    Attributes
    ----------
    success : bool
        True if exploit/scan was successful, False otherwise
    value : float
        value gained from action. Is the value of the host if successfuly
        exploited, otherwise 0
    services : dict
        services identified by action.
    os : dict
        OS identified by action
    processes : dict
        processes identified by action
    access : dict
        access gained by action
    discovered : dict
        host addresses discovered by action
    connection_error : bool
        True if action failed due to connection error (e.g. could
        not reach target)
    permission_error : bool
        True if action failed due to a permission error (e.g. incorrect access
        level to perform action)
    undefined_error : bool
        True if action failed due to an undefined error (e.g. random exploit
        failure)
    newly_discovered : dict
        host addresses discovered for the first time by action
    """

    def __init__(self,
                 success,
                 value=0.0,
                 services=None,
                 os=None,
                 processes=None,
                 access=None,
                 discovered=None,
                 connection_error=False,
                 permission_error=False,
                 undefined_error=False,
                 newly_discovered=None):
        """
        Parameters
        ----------
        success : bool
            True if exploit/scan was successful, False otherwise
        value : float, optional
            value gained from action (default=0.0)
        services : dict, optional
            services identified by action (default=None={})
        os : dict, optional
            OS identified by action (default=None={})
        processes : dict, optional
            processes identified by action (default=None={})
        access : dict, optional
            access gained by action (default=None={})
        discovered : dict, optional
            host addresses discovered by action (default=None={})
        connection_error : bool, optional
            True if action failed due to connection error (default=False)
        permission_error : bool, optional
            True if action failed due to a permission error (default=False)
        undefined_error : bool, optional
            True if action failed due to an undefined error (default=False)
        newly_discovered : dict, optional
            host addresses discovered for first time by action (default=None)
        """
        self.success = success
        self.value = value
        self.services = {} if services is None else services
        self.os = {} if os is None else os
        self.processes = {} if processes is None else processes
        self.access = {} if access is None else access
        self.discovered = {} if discovered is None else discovered
        self.connection_error = connection_error
        self.permission_error = permission_error
        self.undefined_error = undefined_error
        if newly_discovered is not None:
            self.newly_discovered = newly_discovered
        else:
            self.newly_discovered = {}

    def info(self):
        """Get results as dict

        Returns
        -------
        dict
            action results information
        """
        return dict(
            success=self.success,
            value=self.value,
            services=self.services,
            os=self.os,
            processes=self.processes,
            access=self.access,
            discovered=self.discovered,
            connection_error=self.connection_error,
            permission_error=self.permission_error,
            undefined_error=self.undefined_error,
            newly_discovered=self.newly_discovered
        )

    def __str__(self):
        output = ["ActionObservation:"]
        for k, val in self.info().items():
            output.append(f"  {k}={val}")
        return "\n".join(output)


class FlatActionSpace(spaces.Discrete):
    """Không gian hành động phẳng cho môi trường NASim.

    Kế thừa và triển khai không gian hành động gym.spaces.Discrete

    ...

    Attributes
    ----------
    n : int
        số lượng hành động trong không gian hành động
    actions : list of Actions
        danh sách các hành động trong không gian hành động
    """

    def __init__(self, scenario):
        """Khởi tạo không gian hành động phẳng.
        
        Tạo một không gian hành động đơn giản, trong đó mỗi hành động được đại diện 
        bởi một số nguyên từ 0 đến n-1, với n là tổng số hành động có thể thực hiện.
        
        Parameters
        ---------
        scenario : Scenario
            Đối tượng kịch bản mô tả môi trường mạng
        """
        # Tạo danh sách tất cả các hành động có thể thực hiện trong môi trường
        # bao gồm các hành động quét, khai thác lỗ hổng, leo thang đặc quyền cho mỗi máy chủ
        self.actions = load_action_list(scenario)
        
        # Gọi constructor của lớp cha (spaces.Discrete) với số lượng hành động
        # Điều này tạo nên một không gian hành động rời rạc với n phần tử (0 đến n-1)
        super().__init__(len(self.actions))

    def get_action(self, action_idx):
        """Lấy đối tượng Action tương ứng với chỉ số hành động.
        
        Chuyển đổi từ chỉ số hành động (số nguyên) sang đối tượng Action đầy đủ
        có thể được thực hiện trong môi trường.

        Parameters
        ----------
        action_idx : int
            Chỉ số hành động (số nguyên từ 0 đến n-1)

        Returns
        -------
        Action
            Đối tượng Action tương ứng với chỉ số
            
        Raises
        ------
        AssertionError
            Nếu action_idx không phải là số nguyên
        """
        # Kiểm tra nếu action_idx là kiểu dữ liệu numpy (như numpy.int64)
        if hasattr(action_idx, 'dtype'):
            # Chuyển đổi về kiểu int chuẩn của Python
            action_idx = int(action_idx)
            
        # Đảm bảo action_idx là số nguyên
        assert isinstance(action_idx, int), \
            ("Khi sử dụng không gian hành động phẳng, hành động phải là số nguyên"
            f" hoặc đối tượng Action: {action_idx} không hợp lệ")
            
        # Trả về đối tượng Action tương ứng từ danh sách actions
        return self.actions[action_idx]


class ParameterisedActionSpace(spaces.MultiDiscrete):
    """Không gian hành động tham số hóa cho môi trường NASim.
    
    Kế thừa và triển khai gymnasium.spaces.MultiDiscrete, trong đó
    mỗi chiều đại diện cho một tham số hành động khác nhau.
    
    Các tham số hành động (theo thứ tự) là:
    
    0. Loại hành động = [0, 5]
       Trong đó:
         0=Exploit (Khai thác lỗ hổng),
         1=PrivilegeEscalation (Leo thang đặc quyền),
         2=ServiceScan (Quét dịch vụ),
         3=OSScan (Quét hệ điều hành),
         4=SubnetScan (Quét mạng con),
         5=ProcessScan (Quét tiến trình),
         
    1. Subnet = [0, #subnets-1]
       (trừ 1 vì chúng ta không tính subnet internet)
       
    2. Host = [0, max subnets size-1]
    
    3. OS = [0, #OS]
       Trong đó 0=None (không có).
       
    4. Service = [0, #services - 1]
    
    5. Process = [0, #processes]
       Trong đó 0=None (không có).
       
    Lưu ý rằng OS, Service và Process chỉ quan trọng đối với 
    hành động khai thác và leo thang đặc quyền.
    
    ...
    
    Attributes
    ----------
    nvec : Numpy.Array
        Vector chứa kích thước của mỗi tham số
    actions : list of Actions
        Danh sách tất cả các hành động trong không gian hành động
    """

    # Định nghĩa các loại hành động được hỗ trợ
    # Thứ tự này tương ứng với giá trị tham số đầu tiên (action_vec[0])
    action_types = [
        Exploit,               # 0: Khai thác lỗ hổng từ xa
        PrivilegeEscalation,   # 1: Leo thang đặc quyền
        ServiceScan,           # 2: Quét dịch vụ
        OSScan,                # 3: Quét hệ điều hành
        SubnetScan,            # 4: Quét mạng con
        ProcessScan            # 5: Quét tiến trình
    ]

    def __init__(self, scenario):
        """Khởi tạo không gian hành động tham số hóa.
        
        Parameters
        ----------
        scenario : Scenario
            Đối tượng kịch bản mô tả môi trường mạng
        """
        # Lưu trữ kịch bản để sử dụng sau này
        self.scenario = scenario
        
        # Tạo danh sách đầy đủ các hành động có thể thực hiện 
        # (không sử dụng trực tiếp nhưng cần để duy trì tương thích)
        self.actions = load_action_list(scenario)

        # Xác định giới hạn cho từng tham số của vector hành động
        # Mỗi phần tử xác định số lượng giá trị khác nhau cho một tham số
        nvec = [
            len(self.action_types),                # Số loại hành động
            len(self.scenario.subnets)-1,          # Số subnet (trừ internet)
            max(self.scenario.subnets),            # Số host tối đa trong bất kỳ subnet nào
            self.scenario.num_os+1,                # Số OS (+1 cho None)
            self.scenario.num_services,            # Số dịch vụ
            self.scenario.num_processes            # Số tiến trình
        ]

        # Gọi constructor của lớp cha với vector kích thước tham số
        super().__init__(nvec)

    def get_action(self, action_vec):
        """Chuyển đổi vector tham số thành đối tượng Action.
        
        Phương thức này lấy vector tham số (thường do agent trả về) và
        chuyển đổi thành một đối tượng Action có thể được thực hiện
        trong môi trường.

        Parameters
        ----------
        action_vector : list of ints, tuple of ints, hoặc Numpy.Array
            Vector tham số hành động

        Returns
        -------
        Action
            Đối tượng Action tương ứng

        Notes
        -----
        1. Nếu số host được chỉ định trong vector hành động lớn hơn
           số host trong subnet đã chỉ định, thì số host sẽ được tính
           là host# % subnet_size.
        2. Nếu hành động là exploit mà các tham số không khớp với
           bất kỳ định nghĩa exploit nào trong kịch bản, một hành động
           NoOp (không làm gì) sẽ được trả về với chi phí bằng 0.
        """
        # Kiểm tra kiểu dữ liệu của tham số đầu vào
        assert isinstance(action_vec, (list, tuple, np.ndarray)), \
            ("Khi sử dụng không gian hành động tham số hóa, hành động phải là một đối tượng Action, "
             f"một list hoặc một mảng numpy: {action_vec} không hợp lệ")
        
        # Lấy lớp hành động dựa trên tham số đầu tiên
        a_class = self.action_types[action_vec[0]]
        
        # Cần cộng thêm 1 cho subnet để tính đến subnet Internet (thường là subnet 0)
        subnet = action_vec[1]+1
        
        # Đảm bảo host nằm trong giới hạn của subnet được chọn bằng phép toán modulo
        host = action_vec[2] % self.scenario.subnets[subnet]

        # Tạo địa chỉ mục tiêu từ subnet và host
        target = (subnet, host)

        # Xử lý các trường hợp hành động quét (không phải Exploit hoặc PrivilegeEscalation)
        if a_class not in (Exploit, PrivilegeEscalation):
            # Có thể bỏ qua các tham số khác vì không liên quan đến hành động quét
            kwargs = self._get_scan_action_def(a_class)
            # Tạo và trả về đối tượng hành động quét phù hợp
            return a_class(target=target, **kwargs)

        # Xử lý tham số OS: nếu là 0 thì không có OS cụ thể, nếu không thì lấy OS tương ứng
        os = None if action_vec[3] == 0 else self.scenario.os[action_vec[3]-1]

        # Xử lý các trường hợp Exploit (khai thác lỗ hổng)
        if a_class == Exploit:
            # Cần đảm bảo lựa chọn hợp lệ và lấy các tham số cố định (name, cost, prob, access)
            service = self.scenario.services[action_vec[4]]
            # Kiểm tra và lấy định nghĩa exploit cho service và OS cụ thể
            a_def = self._get_exploit_def(service, os)
        else:
            # Xử lý các trường hợp PrivilegeEscalation (leo thang đặc quyền)
            # Cần đảm bảo lựa chọn hợp lệ và lấy các tham số cố định
            proc = self.scenario.processes[action_vec[5]]
            # Kiểm tra và lấy định nghĩa privilege escalation cho process và OS cụ thể
            a_def = self._get_privesc_def(proc, os)

        # Nếu không tìm thấy định nghĩa phù hợp, trả về hành động NoOp (không làm gì)
        if a_def is None:
            return NoOp()
            
        # Tạo và trả về đối tượng Action với các tham số đã lấy được
        return a_class(target=target, **a_def)

    def _get_scan_action_def(self, a_class):
        """Lấy các hằng số cho định nghĩa hành động quét
        
        Phương thức này trả về một từ điển chứa chi phí phù hợp cho
        loại hành động quét dựa trên kịch bản.
        
        Parameters
        ----------
        a_class : class
            Lớp hành động quét (ServiceScan, OSScan, SubnetScan, ProcessScan)
            
        Returns
        -------
        dict
            Từ điển chứa chi phí của hành động quét
            
        Raises
        ------
        TypeError
            Nếu loại hành động không được hỗ trợ
        """
        # Lấy chi phí phù hợp cho từng loại hành động quét
        if a_class == ServiceScan:
            cost = self.scenario.service_scan_cost
        elif a_class == OSScan:
            cost = self.scenario.os_scan_cost
        elif a_class == SubnetScan:
            cost = self.scenario.subnet_scan_cost
        elif a_class == ProcessScan:
            cost = self.scenario.process_scan_cost
        else:
            # Báo lỗi nếu loại hành động không được hỗ trợ
            raise TypeError(f"Không được triển khai cho lớp hành động {a_class}")
            
        # Trả về từ điển với chi phí
        return {"cost": cost}

    def _get_exploit_def(self, service, os):
        """Kiểm tra xem tham số exploit có hợp lệ không
        
        Phương thức này kiểm tra xem có tồn tại exploit cho dịch vụ 
        và hệ điều hành được chỉ định hay không.
        
        Parameters
        ----------
        service : str
            Tên dịch vụ mục tiêu
        os : str
            Tên hệ điều hành mục tiêu
            
        Returns
        -------
        dict or None
            Nếu tồn tại exploit phù hợp, trả về từ điển chứa thông tin exploit
            Nếu không, trả về None
        """
        # Lấy bảng ánh xạ exploit từ kịch bản
        e_map = self.scenario.exploit_map
        
        # Kiểm tra xem dịch vụ có nằm trong bảng ánh xạ không
        if service not in e_map:
            return None
            
        # Kiểm tra xem OS có nằm trong bảng ánh xạ của dịch vụ không
        if os not in e_map[service]:
            return None
            
        # Nếu có, trả về thông tin định nghĩa exploit
        return e_map[service][os]

    def _get_privesc_def(self, proc, os):
        """Kiểm tra xem tham số leo thang đặc quyền có hợp lệ không
        
        Phương thức này kiểm tra xem có tồn tại kỹ thuật leo thang đặc quyền
        cho tiến trình và hệ điều hành được chỉ định hay không.
        
        Parameters
        ----------
        proc : str
            Tên tiến trình mục tiêu
        os : str
            Tên hệ điều hành mục tiêu
            
        Returns
        -------
        dict or None
            Nếu tồn tại kỹ thuật leo thang đặc quyền phù hợp, trả về từ điển chứa thông tin
            Nếu không, trả về None
        """
        # Lấy bảng ánh xạ privilege escalation từ kịch bản
        pe_map = self.scenario.privesc_map
        
        # Kiểm tra xem tiến trình có nằm trong bảng ánh xạ không
        if proc not in pe_map:
            return None
            
        # Kiểm tra xem OS có nằm trong bảng ánh xạ của tiến trình không
        if os not in pe_map[proc]:
            return None
            
        # Nếu có, trả về thông tin định nghĩa privilege escalation
        return pe_map[proc][os]