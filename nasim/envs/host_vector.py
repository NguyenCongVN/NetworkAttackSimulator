""" This module contains the HostVector class.

This is the main class for storing and updating the state of a single host
in the NASim environment.
"""

import numpy as np

from nasim.envs.utils import AccessLevel
from nasim.envs.action import ActionResult
from logger import logger


class HostVector:
    """ A Vector representation of a single host in NASim.

    Each host is represented as a vector (1D numpy array) for efficiency and to
    make it easier to use with deep learning agents. The vector is made up of
    multiple features arranged in a consistent way.

    Features in the vector, listed in order, are:

    1. subnet address - one-hot encoding with length equal to the number
                        of subnets
    2. host address - one-hot encoding with length equal to the maximum number
                      of hosts in any subnet
    3. compromised - bool
    4. reachable - bool
    5. discovered - bool
    6. value - float
    7. discovery value - float
    8. access - int
    9. OS - bool for each OS in scenario (only one OS has value of true)
    10. services running - bool for each service in scenario
    11. processes running - bool for each process in scenario

    Notes
    -----
    - The size of the vector is equal to:

        #subnets + max #hosts in any subnet + 6 + #OS + #services + #processes.

    - Where the +6 is for compromised, reachable, discovered, value,
      discovery_value, and access features
    - The vector is a float vector so True/False is actually represented as
      1.0/0.0.

    """

    # class properties that are the same for all hosts
    # these are set when calling vectorize method
    # the bounds on address space (used for one hot encoding of host address)
    address_space_bounds = None
    # number of OS in scenario
    num_os = None
    # map from OS name to its index in host vector
    os_idx_map = {}
    # number of services in scenario
    num_services = None
    # map from service name to its index in host vector
    service_idx_map = {}
    # number of processes in scenario
    num_processes = None
    # map from process name to its index in host vector
    process_idx_map = {}
    # size of state for host vector (i.e. len of vector)
    state_size = None

    # vector position constants
    # to be initialized
    _subnet_address_idx = 0
    _host_address_idx = None
    _compromised_idx = None
    _reachable_idx = None
    _discovered_idx = None
    _value_idx = None
    _discovery_value_idx = None
    _access_idx = None
    _os_start_idx = None
    _service_start_idx = None
    _process_start_idx = None

    def __init__(self, vector):
        self.vector = vector

    @classmethod
    def vectorize(cls, host, address_space_bounds, vector=None):
        """Chuyển đổi thông tin của một host thành vector số học.
        
        Phương thức này chuyển đổi các thuộc tính của một host (như địa chỉ, trạng thái, các dịch vụ, 
        hệ điều hành và quy trình) thành một vector số học để sử dụng trong mô hình học máy.
        
        Parameters:
        -----------
        host : Host
            Đối tượng host cần chuyển đổi thành vector.
        address_space_bounds : tuple
            Giới hạn của không gian địa chỉ (số lượng subnet và host trong mỗi subnet).
        vector : numpy.ndarray, optional
            Vector đã có sẵn để cập nhật. Nếu không được cung cấp, một vector mới sẽ được tạo.
            
        Returns:
        --------
        HostVector
            Một instance của lớp HostVector chứa thông tin của host dưới dạng vector.
            
        Notes:
        ------
        Vector kết quả có cấu trúc như sau:
        - Các phần tử đầu tiên biểu diễn địa chỉ subnet (one-hot encoding)
        - Tiếp theo là địa chỉ host (one-hot encoding)
        - Sau đó là các trạng thái: compromised, reachable, discovered
        - Tiếp theo là giá trị của host, giá trị khi phát hiện và mức độ truy cập
        - Cuối cùng là thông tin về hệ điều hành, dịch vụ và quy trình
        """
        # Kiểm tra xem đã khởi tạo các thông số cơ bản chưa
        if cls.address_space_bounds is None:
            # Nếu chưa khởi tạo, gọi phương thức _initialize để thiết lập 
            # các thông số về giới hạn không gian địa chỉ, ánh xạ OS, dịch vụ và tiến trình
            # Đây là bước cần thiết trước khi chuyển đổi thông tin host thành vector
            cls._initialize(
                address_space_bounds, host.services, host.os, host.processes
            )
    
        # Tạo vector mới hoặc sử dụng vector đã được cung cấp
        if vector is None:
            # Nếu không có vector, tạo vector mới với kích thước state_size và giá trị 0
            vector = np.zeros(cls.state_size, dtype=np.float32)
        else:
            # Nếu vector được cung cấp, kiểm tra kích thước để đảm bảo tính nhất quán
            assert len(vector) == cls.state_size
    
        # Điền thông tin địa chỉ vào vector sử dụng mã hóa one-hot
        # Subnet address: đặt giá trị 1 tại vị trí tương ứng với địa chỉ subnet của host
        vector[cls._subnet_address_idx + host.address[0]] = 1
        # Host address: đặt giá trị 1 tại vị trí tương ứng với địa chỉ host trong subnet
        vector[cls._host_address_idx + host.address[1]] = 1
        
        # Điền các thông tin trạng thái cơ bản của host
        vector[cls._compromised_idx] = int(host.compromised)  # Trạng thái đã bị xâm nhập
        vector[cls._reachable_idx] = int(host.reachable)      # Trạng thái có thể tiếp cận
        vector[cls._discovered_idx] = int(host.discovered)    # Trạng thái đã được khám phá
        
        # Điền thông tin về giá trị và mức truy cập
        vector[cls._value_idx] = host.value                   # Giá trị khi chiếm quyền root
        vector[cls._discovery_value_idx] = host.discovery_value  # Giá trị khi phát hiện
        vector[cls._access_idx] = host.access                 # Mức độ quyền truy cập
        
        # Điền thông tin về hệ điều hành
        for os_num, (os_key, os_val) in enumerate(host.os.items()):
            # _get_os_idx trả về vị trí trong vector cho hệ điều hành dựa trên os_num
            vector[cls._get_os_idx(os_num)] = int(os_val)  # Chuyển boolean thành 0/1
        
        # Điền thông tin về các dịch vụ đang chạy
        for srv_num, (srv_key, srv_val) in enumerate(host.services.items()):
            # _get_service_idx trả về vị trí trong vector cho dịch vụ dựa trên srv_num
            vector[cls._get_service_idx(srv_num)] = int(srv_val)  # Chuyển boolean thành 0/1
        
        # Điền thông tin về các tiến trình đang chạy
        host_procs = host.processes.items()
        for proc_num, (proc_key, proc_val) in enumerate(host_procs):
            # _get_process_idx trả về vị trí trong vector cho tiến trình dựa trên proc_num
            vector[cls._get_process_idx(proc_num)] = int(proc_val)  # Chuyển boolean thành 0/1
        
        # Tạo và trả về đối tượng HostVector mới với vector đã điền thông tin
        return cls(vector)

    @classmethod
    def vectorize_random(cls, host, address_space_bounds, vector=None):
        """Tạo một vector host với các dịch vụ, hệ điều hành và tiến trình được thiết lập ngẫu nhiên.
        
        Phương thức này đầu tiên tạo một vector host bình thường từ thông tin của host cung cấp,
        sau đó ghi đè các thông tin về dịch vụ, hệ điều hành, và tiến trình bằng các giá trị ngẫu nhiên.
        Điều này hữu ích để tạo các host ngẫu nhiên cho việc thử nghiệm hoặc khởi tạo mạng đa dạng.
        
        Parameters
        ----------
        host : Host
            Đối tượng host cơ sở để lấy thông tin cơ bản (như địa chỉ, trạng thái xâm nhập...)
        address_space_bounds : tuple
            Giới hạn của không gian địa chỉ (số lượng subnet và host trong mỗi subnet)
        vector : numpy.ndarray, optional
            Vector đã có sẵn để cập nhật. Nếu không có, một vector mới sẽ được tạo
            
        Returns
        -------
        HostVector
            Đối tượng HostVector với các thuộc tính dịch vụ, hệ điều hành và tiến trình được thiết lập ngẫu nhiên
            
        Note
        ----
        Phương thức này giữ nguyên các thuộc tính cơ bản của host như địa chỉ, trạng thái xâm nhập,
        khả năng tiếp cận, giá trị, và quyền truy cập, chỉ thay đổi các thông tin về cấu hình kỹ thuật.
        """
        # Tạo vector host cơ bản từ thông tin host đã cung cấp
        hvec = cls.vectorize(host, address_space_bounds, vector)
        
        # Thiết lập ngẫu nhiên các dịch vụ
        for srv_num in cls.service_idx_map.values():
            # Tạo giá trị ngẫu nhiên 0 hoặc 1 (False/True) cho từng dịch vụ
            srv_val = np.random.randint(0, 2)
            # Cập nhật trạng thái dịch vụ trong vector
            hvec.vector[cls._get_service_idx(srv_num)] = srv_val

        # Thiết lập ngẫu nhiên hệ điều hành
        # Chỉ một hệ điều hành được chọn ngẫu nhiên (chỉ một OS có thể chạy trên một host)
        chosen_os = np.random.choice(list(cls.os_idx_map.values()))
        for os_num in cls.os_idx_map.values():
            # Đặt giá trị 1 cho OS được chọn, 0 cho các OS khác
            # int(os_num == chosen_os) chuyển đổi biểu thức boolean thành 0/1
            hvec.vector[cls._get_os_idx(os_num)] = int(os_num == chosen_os)

        # Thiết lập ngẫu nhiên các tiến trình
        for proc_num in cls.process_idx_map.values():
            # Tạo giá trị ngẫu nhiên 0 hoặc 1 (False/True) cho từng tiến trình
            proc_val = np.random.randint(0, 2)
            # Cập nhật trạng thái tiến trình trong vector
            hvec.vector[cls._get_process_idx(proc_num)] = proc_val
        
        return hvec

    @property
    def compromised(self):
        return self.vector[self._compromised_idx]

    @compromised.setter
    def compromised(self, val):
        self.vector[self._compromised_idx] = int(val)

    @property
    def discovered(self):
        return self.vector[self._discovered_idx]

    @discovered.setter
    def discovered(self, val):
        self.vector[self._discovered_idx] = int(val)

    @property
    def reachable(self):
        return self.vector[self._reachable_idx]

    @reachable.setter
    def reachable(self, val):
        self.vector[self._reachable_idx] = int(val)

    @property
    def address(self):
        return (
            self.vector[self._subnet_address_idx_slice()].argmax(),
            self.vector[self._host_address_idx_slice()].argmax()
        )

    @property
    def value(self):
        return self.vector[self._value_idx]

    @property
    def discovery_value(self):
        return self.vector[self._discovery_value_idx]

    @property
    def access(self):
        return self.vector[self._access_idx]

    @access.setter
    def access(self, val):
        self.vector[self._access_idx] = int(val)

    @property
    def services(self):
        services = {}
        for srv, srv_num in self.service_idx_map.items():
            services[srv] = self.vector[self._get_service_idx(srv_num)]
        return services

    @property
    def os(self):
        os = {}
        for os_key, os_num in self.os_idx_map.items():
            os[os_key] = self.vector[self._get_os_idx(os_num)]
        return os

    @property
    def processes(self):
        processes = {}
        for proc, proc_num in self.process_idx_map.items():
            processes[proc] = self.vector[self._get_process_idx(proc_num)]
        return processes

    @property
    def monitoring(self):
        """Trạng thái giám sát của host (1.0 = đang được giám sát, 0.0 = không giám sát)"""
        return self.vector[self._monitoring_idx]

    @monitoring.setter
    def monitoring(self, val):
        """Thiết lập trạng thái giám sát cho host"""
        self.vector[self._monitoring_idx] = float(bool(val))

    @property
    def decoy_services(self):
        """Lấy thông tin về các dịch vụ giả mạo đang chạy trên host dưới dạng từ điển"""
        decoys = {}
        # Sử dụng cùng tên dịch vụ như services thực, có thể điều chỉnh tùy nhu cầu
        for srv, srv_num in self.service_idx_map.items():
            decoys[srv] = bool(self.vector[self._decoy_services_start_idx + srv_num])
        return decoys

    def set_decoy_service(self, service_name, value=True):
        """Thiết lập một dịch vụ giả mạo cụ thể trên host
        
        Parameters
        ----------
        service_name : str
            Tên dịch vụ giả mạo cần thiết lập
        value : bool, mặc định=True
            True để bật dịch vụ giả mạo, False để tắt
        """
        if service_name in self.service_idx_map:
            srv_num = self.service_idx_map[service_name]
            self.vector[self._decoy_services_start_idx + srv_num] = float(bool(value))
        else:
            raise KeyError(f"Dịch vụ '{service_name}' không tồn tại trong service_idx_map")
        
    @property
    def confident(self):
        """Lấy mức độ tin cậy của host dưới dạng từ điển"""
        # Các mức tin cậy: Low, Medium, High
        confidence_levels = ["Low", "Medium", "High"]
        confidence = {}
        for i, level in enumerate(confidence_levels):
            confidence[level] = bool(self.vector[self._confident_start_idx + i])
        return confidence

    def set_confident_level(self, level):
        """Thiết lập mức độ tin cậy cho host
        
        Parameters
        ----------
        level : str
            Mức độ tin cậy ("Low", "Medium", hoặc "High")
        """
        confidence_levels = ["Low", "Medium", "High"]
        if level in confidence_levels:
            # Reset tất cả các mức tin cậy về 0
            for i in range(self.num_confident_levels):
                self.vector[self._confident_start_idx + i] = 0.0
            # Đặt mức tin cậy được chọn thành 1
            level_idx = confidence_levels.index(level)
            self.vector[self._confident_start_idx + level_idx] = 1.0
        else:
            raise ValueError(f"Mức độ tin cậy không hợp lệ. Phải là một trong: {confidence_levels}")

    def is_running_service(self, srv):
        """Kiểm tra xem dịch vụ cụ thể có đang chạy trên host hay không.
        
        Phương thức này sử dụng mapping từ tên dịch vụ đến vị trí của nó trong vector
        để kiểm tra trạng thái của dịch vụ đó (đang chạy hoặc không chạy).
        
        Parameters
        ----------
        srv : str
            Tên dịch vụ cần kiểm tra (ví dụ: "HTTP", "SSH", "FTP")
            
        Returns
        -------
        bool
            True nếu dịch vụ đang chạy trên host, False nếu không
            
        Raises
        ------
        KeyError
            Nếu tên dịch vụ không tồn tại trong service_idx_map
            
        Example
        -------
        >>> host.is_running_service("HTTP")
        True  # Nếu dịch vụ HTTP đang chạy trên host
        """
        # Lấy chỉ số của dịch vụ trong ánh xạ dịch vụ
        srv_num = self.service_idx_map[srv]
        # Tính vị trí của dịch vụ trong vector và chuyển đổi giá trị thành boolean
        return bool(self.vector[self._get_service_idx(srv_num)])
    
    def is_running_os(self, os):
        """Kiểm tra xem hệ điều hành cụ thể có đang chạy trên host hay không.
        
        Phương thức này sử dụng mapping từ tên hệ điều hành đến vị trí của nó trong vector
        để kiểm tra xem host có đang chạy hệ điều hành đó hay không.
        
        Parameters
        ----------
        os : str
            Tên hệ điều hành cần kiểm tra (ví dụ: "Windows", "Linux")
            
        Returns
        -------
        bool
            True nếu host đang chạy hệ điều hành đã chỉ định, False nếu không
            
        Raises
        ------
        KeyError
            Nếu tên hệ điều hành không tồn tại trong os_idx_map
            
        Note
        ----
        Mỗi host chỉ có thể chạy một hệ điều hành tại một thời điểm, vì vậy
        nếu gọi is_running_os với nhiều hệ điều hành khác nhau, chỉ một trong số đó
        sẽ trả về True.
        """
        # Lấy chỉ số của hệ điều hành trong ánh xạ OS
        os_num = self.os_idx_map[os]
        # Tính vị trí của hệ điều hành trong vector và chuyển đổi giá trị thành boolean
        return bool(self.vector[self._get_os_idx(os_num)])
    
    def is_running_process(self, proc):
        """Kiểm tra xem một tiến trình cụ thể có đang chạy trên host hay không.
        
        Phương thức này sử dụng mapping từ tên tiến trình đến vị trí của nó trong vector
        để kiểm tra trạng thái của tiến trình đó (đang chạy hoặc không chạy).
        
        Parameters
        ----------
        proc : str
            Tên tiến trình cần kiểm tra (ví dụ: "Apache", "MySQL")
            
        Returns
        -------
        bool
            True nếu tiến trình đang chạy trên host, False nếu không
            
        Raises
        ------
        KeyError
            Nếu tên tiến trình không tồn tại trong process_idx_map
            
        Note
        ----
        Tiến trình được sử dụng chủ yếu trong các hành động escalate_privilege (leo thang đặc quyền),
        nơi một số tiến trình cụ thể có thể chứa lỗ hổng cho phép nâng cao quyền truy cập.
        """
        # Lấy chỉ số của tiến trình trong ánh xạ process
        proc_num = self.process_idx_map[proc]
        # Tính vị trí của tiến trình trong vector và chuyển đổi giá trị thành boolean
        return bool(self.vector[self._get_process_idx(proc_num)])

    def perform_action(self, action, DEBUG=False):
        """Perform given action against this host
    
        Arguments
        ---------
        action : Action
            the action to perform
        DEBUG : bool
            whether to print debug information
    
        Returns
        -------
        HostVector
            the resulting state of host after action
        ActionObservation
            the result from the action
        """
        # Lưu tên hàm để sử dụng trong các thông báo debug
        function_name = "perform_action"
        
        # In thông tin debug về hành động đang được thực hiện
        if DEBUG:
            logger.debug(f"[DEBUG - {function_name}] Thực hiện hành động: {action.name} trên host: {self.address}")
        
        next_state = self.copy()
        if action.is_service_scan():
            if DEBUG:
                logger.debug(f"[DEBUG - {function_name}] Quét dịch vụ thành công, tìm thấy: {list(self.services.keys())}")
            result = ActionResult(True, 0, services=self.services)
            return next_state, result
    
        if action.is_os_scan():
            if DEBUG:
                logger.debug(f"[DEBUG - {function_name}] Quét hệ điều hành thành công, tìm thấy: {list(self.os.keys())}")
            return next_state, ActionResult(True, 0, os=self.os)
    
        if action.is_exploit():
            if DEBUG:
                logger.debug(f"[DEBUG - {function_name}] Đang khai thác dịch vụ: {action.service}" + 
                      (f" trên OS: {action.os}" if action.os else ""))
            
            if self.is_running_service(action.service) and \
               (action.os is None or self.is_running_os(action.os)):
                # service and os is present so exploit is successful
                value = 0
                next_state.compromised = True
                if not self.access == AccessLevel.ROOT:
                    # ensure a machine is not rewarded twice
                    # and access doesn't decrease
                    next_state.access = action.access
                    if action.access == AccessLevel.ROOT:
                        value = self.value
                        if DEBUG:
                            logger.debug(f"[DEBUG - {function_name}] Khai thác thành công! Đạt quyền ROOT - nhận giá trị: {value}")
                    elif DEBUG:
                        logger.debug(f"[DEBUG - {function_name}] Khai thác thành công! Nâng quyền từ {self.access} lên {action.access}")
    
                result = ActionResult(
                    True,
                    value=value,
                    services=self.services,
                    os=self.os,
                    access=action.access
                )
                return next_state, result
            elif DEBUG:
                if not self.is_running_service(action.service):
                    logger.debug(f"[DEBUG - {function_name}] Khai thác thất bại - dịch vụ {action.service} không chạy trên host")
                elif action.os is not None and not self.is_running_os(action.os):
                    logger.debug(f"[DEBUG - {function_name}] Khai thác thất bại - host không chạy OS {action.os}")
    
        # following actions are on host so require correct access
        if not (self.compromised and action.req_access <= self.access):
            if DEBUG:
                if not self.compromised:
                    logger.debug(f"[DEBUG - {function_name}] Hành động thất bại - host chưa bị xâm nhập")
                else:
                    logger.debug(f"[DEBUG - {function_name}] Hành động thất bại - quyền không đủ (cần {action.req_access}, hiện tại: {self.access})")
            result = ActionResult(False, 0, permission_error=True)
            return next_state, result
    
        if action.is_process_scan():
            if DEBUG:
                logger.debug(f"[DEBUG - {function_name}] Quét tiến trình thành công, tìm thấy: {list(self.processes.keys())}")
            result = ActionResult(
                True, 0, access=self.access, processes=self.processes
            )
            return next_state, result
    
        if action.is_privilege_escalation():
            if DEBUG:
                logger.debug(f"[DEBUG - {function_name}] Đang leo thang đặc quyền" + 
                      (f" sử dụng tiến trình: {action.process}" if action.process else ""))
            
            has_proc = (
                action.process is None
                or self.is_running_process(action.process)
            )
            has_os = (
                action.os is None or self.is_running_os(action.os)
            )
            if has_proc and has_os:
                # host compromised and proc and os is present
                # so privesc is successful
                value = 0.0
                if not self.access == AccessLevel.ROOT:
                    # ensure a machine is not rewarded twice
                    # and access doesn't decrease
                    next_state.access = action.access
                    if action.access == AccessLevel.ROOT:
                        value = self.value
                        if DEBUG:
                            logger.debug(f"[DEBUG - {function_name}] Leo thang thành công! Đạt quyền ROOT - nhận giá trị: {value}")
                    elif DEBUG:
                        logger.debug(f"[DEBUG - {function_name}] Leo thang thành công! Nâng quyền từ {self.access} lên {action.access}")
                    
                result = ActionResult(
                    True,
                    value=value,
                    processes=self.processes,
                    os=self.os,
                    access=action.access
                )
                return next_state, result
            elif DEBUG:
                if not has_proc:
                    logger.debug(f"[DEBUG - {function_name}] Leo thang thất bại - tiến trình {action.process} không có mặt")
                elif not has_os:
                    logger.debug(f"[DEBUG - {function_name}] Leo thang thất bại - OS không phù hợp")
    
        # action failed due to host config not meeting preconditions
        if DEBUG:
            logger.debug(f"[DEBUG - {function_name}] Hành động thất bại - host không đáp ứng điều kiện cần thiết")
        return next_state, ActionResult(False, 0)

    def observe(self,
                address=False,
                compromised=False,
                reachable=False,
                discovered=False,
                access=False,
                value=False,
                discovery_value=False,
                services=False,
                processes=False,
                os=False):
        """Tạo một vector quan sát (observation) cho host với thông tin được chọn.
        
        Phương thức này tạo ra một vector quan sát chỉ chứa những thông tin được yêu cầu
        thông qua các tham số boolean. Điều này cho phép tạo ra các quan sát một phần
        về host, tuỳ thuộc vào những gì agent được phép biết hoặc đã khám phá được.
        
        Parameters
        ----------
        address : bool, mặc định=False
            Nếu True, bao gồm thông tin địa chỉ subnet và host (dạng one-hot encoding)
        compromised : bool, mặc định=False 
            Nếu True, bao gồm trạng thái xâm nhập (đã bị tấn công thành công chưa)
        reachable : bool, mặc định=False
            Nếu True, bao gồm trạng thái có thể tiếp cận (có thể kết nối tới host này không)
        discovered : bool, mặc định=False
            Nếu True, bao gồm trạng thái đã được khám phá (agent đã biết về host này chưa)
        access : bool, mặc định=False
            Nếu True, bao gồm cấp độ quyền truy cập hiện tại (USER, ADMIN, ROOT...)
        value : bool, mặc định=False
            Nếu True, bao gồm giá trị thưởng khi chiếm được host này
        discovery_value : bool, mặc định=False
            Nếu True, bao gồm giá trị thưởng khi khám phá ra host này
        services : bool, mặc định=False
            Nếu True, bao gồm thông tin về các dịch vụ đang chạy (HTTP, SSH...)
        processes : bool, mặc định=False
            Nếu True, bao gồm thông tin về các tiến trình đang chạy
        os : bool, mặc định=False
            Nếu True, bao gồm thông tin về hệ điều hành của host
            
        Returns
        -------
        numpy.ndarray
            Vector quan sát chỉ chứa thông tin được yêu cầu, các phần còn lại là 0
        """
        # Khởi tạo vector quan sát trống với cùng kích thước như vector trạng thái đầy đủ
        obs = np.zeros(self.state_size, dtype=np.float32)
        
        # Chỉ điền thông tin địa chỉ khi được yêu cầu
        if address:
            # Lấy các slice (phần cắt) của vector tương ứng với thông tin địa chỉ subnet và host
            subnet_slice = self._subnet_address_idx_slice()
            host_slice = self._host_address_idx_slice()
            # Sao chép thông tin địa chỉ từ vector gốc sang vector quan sát
            obs[subnet_slice] = self.vector[subnet_slice]  # Địa chỉ subnet (one-hot encoding)
            obs[host_slice] = self.vector[host_slice]      # Địa chỉ host (one-hot encoding)
        
        # Thêm thông tin về trạng thái xâm nhập nếu được yêu cầu
        if compromised:
            obs[self._compromised_idx] = self.vector[self._compromised_idx]
        
        # Thêm thông tin về khả năng tiếp cận nếu được yêu cầu
        if reachable:
            obs[self._reachable_idx] = self.vector[self._reachable_idx]
        
        # Thêm thông tin về trạng thái đã được khám phá nếu được yêu cầu
        if discovered:
            obs[self._discovered_idx] = self.vector[self._discovered_idx]
        
        # Thêm thông tin về giá trị thưởng khi chiếm được host nếu được yêu cầu
        if value:
            obs[self._value_idx] = self.vector[self._value_idx]
        
        # Thêm thông tin về giá trị thưởng khi khám phá ra host nếu được yêu cầu
        if discovery_value:
            v = self.vector[self._discovery_value_idx]
            obs[self._discovery_value_idx] = v
        
        # Thêm thông tin về cấp độ quyền truy cập nếu được yêu cầu
        if access:
            obs[self._access_idx] = self.vector[self._access_idx]
        
        # Thêm thông tin về hệ điều hành nếu được yêu cầu
        if os:
            # Lấy slice của vector tương ứng với tất cả thông tin OS
            idxs = self._os_idx_slice()
            obs[idxs] = self.vector[idxs]  # Sao chép toàn bộ thông tin OS
        
        # Thêm thông tin về các dịch vụ đang chạy nếu được yêu cầu
        if services:
            # Lấy slice của vector tương ứng với tất cả thông tin dịch vụ
            idxs = self._service_idx_slice()
            obs[idxs] = self.vector[idxs]  # Sao chép toàn bộ thông tin dịch vụ
        
        # Thêm thông tin về các tiến trình đang chạy nếu được yêu cầu
        if processes:
            # Lấy slice của vector tương ứng với tất cả thông tin tiến trình
            idxs = self._process_idx_slice()
            obs[idxs] = self.vector[idxs]  # Sao chép toàn bộ thông tin tiến trình
        
        # Trả về vector quan sát đã được điền thông tin theo yêu cầu
        return obs

    def readable(self):
        return self.get_readable(self.vector)

    def copy(self):
        vector_copy = np.copy(self.vector)
        return HostVector(vector_copy)

    def numpy(self):
        return self.vector

    @classmethod
    def _initialize(cls, address_space_bounds, services, os_info, processes):
        """Khởi tạo các thông tin cơ bản cho vector host.
        
        Phương thức này thiết lập các ánh xạ (mapping) giữa các hệ điều hành, dịch vụ, 
        và tiến trình với các chỉ số tương ứng trong vector. Mỗi thành phần của host được lưu ở vị trí nhất định trong vector
        Nó cũng lưu trữ thông tin về giới hạn không gian địa chỉ.
        
        Args:
            address_space_bounds (tuple): Giới hạn không gian địa chỉ (thường là kích thước mạng)
            services (dict): Từ điển chứa thông tin về các dịch vụ có sẵn
            os_info (dict): Từ điển chứa thông tin về các hệ điều hành có sẵn
            processes (dict): Từ điển chứa thông tin về các tiến trình có sẵn
        
        Note:
            - Các ánh xạ được lưu trữ dưới dạng các từ điển trong các biến lớp
            - Phương thức này gọi _update_vector_idxs() để cập nhật các chỉ số vector
            
        Exmample:
            * Tạo một host vector
            address_space_bounds = (2, 3)  * 2 subnet, mỗi subnet tối đa 3 host
            services = {"HTTP": True, "SSH": True, "FTP": True}
            os_info = {"Windows": True, "Linux": True}
            processes = {"Apache": True, "MySQL": True}

            * Gọi _initialize
            HostVector._initialize(address_space_bounds, services, os_info, processes)

            * Kết quả sau khi gọi _initialize và _update_vector_idxs
            * cls.num_os = 2
            * cls.num_services = 3
            * cls.num_processes = 2

            * Vị trí trong vector (sau khi _update_vector_idxs được gọi):
            * cls._subnet_address_idx = 0  * Vị trí bắt đầu cho subnet (2 phần tử)
            * cls._host_address_idx = 2    * Vị trí bắt đầu cho host (3 phần tử) 
            * cls._compromised_idx = 5     * Vị trí cho trạng thái xâm nhập
            * ...
            * cls._os_start_idx = 11       * Vị trí bắt đầu cho OS (2 phần tử)
            * cls._service_start_idx = 13  * Vị trí bắt đầu cho services (3 phần tử)
            * cls._process_start_idx = 16  * Vị trí bắt đầu cho processes (2 phần tử)
            * cls.state_size = 18          * Tổng kích thước vector

            * Ánh xạ (mapping) được tạo ra:
            * cls.os_idx_map = {"Windows": 0, "Linux": 1}
            * cls.service_idx_map = {"HTTP": 0, "SSH": 1, "FTP": 2}
            * cls.process_idx_map = {"Apache": 0, "MySQL": 1}
        """
        cls.os_idx_map = {}
        cls.service_idx_map = {}
        cls.process_idx_map = {}
        cls.address_space_bounds = address_space_bounds
        cls.num_os = len(os_info)
        cls.num_services = len(services)
        cls.num_processes = len(processes)
        cls._update_vector_idxs()
        for os_num, (os_key, os_val) in enumerate(os_info.items()):
            cls.os_idx_map[os_key] = os_num
        for srv_num, (srv_key, srv_val) in enumerate(services.items()):
            cls.service_idx_map[srv_key] = srv_num
        for proc_num, (proc_key, proc_val) in enumerate(processes.items()):
            cls.process_idx_map[proc_key] = proc_num

    @classmethod
    def _update_vector_idxs(cls):
        """Cập nhật các chỉ số vị trí trong vector trạng thái của host.

        Phương thức này thiết lập các chỉ số vị trí cho từng thành phần trong vector 
        trạng thái của host, bao gồm địa chỉ subnet, địa chỉ host, trạng thái xâm nhập,
        khả năng tiếp cận, trạng thái phát hiện, giá trị, giá trị khi phát hiện,
        quyền truy cập, hệ điều hành, dịch vụ và quy trình.

        Các chỉ số này được sử dụng để xác định vị trí của mỗi thành phần trong vector
        trạng thái, giúp truy xuất và cập nhật thông tin trạng thái một cách hiệu quả.

        Chú ý:
            - Đây là một phương thức lớp (class method) được gọi khi khởi tạo lớp
            - Kết quả cuối cùng xác định kích thước của vector trạng thái (state_size)
            - Các chỉ số được tính toán dựa trên các giới hạn không gian địa chỉ đã định nghĩa
        """
        cls._subnet_address_idx = 0
        cls._host_address_idx = cls.address_space_bounds[0]
        cls._compromised_idx = (
            cls._host_address_idx + cls.address_space_bounds[1]
        )
        cls._reachable_idx = cls._compromised_idx + 1
        cls._discovered_idx = cls._reachable_idx + 1
        cls._value_idx = cls._discovered_idx + 1
        cls._discovery_value_idx = cls._value_idx + 1
        cls._access_idx = cls._discovery_value_idx + 1
        cls._os_start_idx = cls._access_idx + 1
        cls._service_start_idx = cls._os_start_idx + cls.num_os
        cls._process_start_idx = cls._service_start_idx + cls.num_services
        cls.state_size = cls._process_start_idx + cls.num_processes
        
        # Định nghĩa các chỉ số cụ thể
        cls._monitoring_idx = cls._process_start_idx + cls.num_processes  # Chỉ số cho trạng thái giám sát (bool)
        cls._decoy_services_start_idx = cls._monitoring_idx + 1  # Vị trí bắt đầu của decoy services (one-hot)
        cls._confident_start_idx = cls._decoy_services_start_idx + cls.num_services  # Vị trí bắt đầu của confident (one-hot)
        
        # Giả sử confident có 3 mức: Low, Medium, High
        cls.num_confident_levels = 3
        
        # Cập nhật tổng kích thước vector
        cls.state_size = cls._confident_start_idx + cls.num_confident_levels

    @classmethod
    def _subnet_address_idx_slice(cls):
        return slice(cls._subnet_address_idx, cls._host_address_idx)

    @classmethod
    def _host_address_idx_slice(cls):
        return slice(cls._host_address_idx, cls._compromised_idx)

    @classmethod
    def _get_service_idx(cls, srv_num):
        return cls._service_start_idx+srv_num

    @classmethod
    def _service_idx_slice(cls):
        return slice(cls._service_start_idx, cls._process_start_idx)

    @classmethod
    def _get_os_idx(cls, os_num):
        return cls._os_start_idx+os_num

    @classmethod
    def _os_idx_slice(cls):
        return slice(cls._os_start_idx, cls._service_start_idx)

    @classmethod
    def _get_process_idx(cls, proc_num):
        return cls._process_start_idx+proc_num

    @classmethod
    def _process_idx_slice(cls):
        return slice(cls._process_start_idx, cls.state_size)

    @classmethod
    def get_readable(cls, vector):
        readable_dict = dict()
        hvec = cls(vector)
        readable_dict["Address"] = hvec.address
        readable_dict["Compromised"] = bool(hvec.compromised)
        readable_dict["Reachable"] = bool(hvec.reachable)
        readable_dict["Discovered"] = bool(hvec.discovered)
        readable_dict["Value"] = hvec.value
        readable_dict["Discovery Value"] = hvec.discovery_value
        readable_dict["Access"] = hvec.access
        for os_name in cls.os_idx_map:
            readable_dict[f"{os_name}"] = hvec.is_running_os(os_name)
        for srv_name in cls.service_idx_map:
            readable_dict[f"{srv_name}"] = hvec.is_running_service(srv_name)
        for proc_name in cls.process_idx_map:
            readable_dict[f"{proc_name}"] = hvec.is_running_process(proc_name)

        return readable_dict

    @classmethod
    def reset(cls):
        """Resets any class variables.

        This is used to avoid errors when changing scenarios within a single
        python session
        """
        cls.address_space_bounds = None

    def __repr__(self):
        return f"Host: {self.address}"

    def __hash__(self):
        return hash(str(self.vector))

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, HostVector):
            return False
        return np.array_equal(self.vector, other.vector)
