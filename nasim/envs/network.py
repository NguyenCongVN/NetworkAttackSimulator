import numpy as np

from nasim.envs.action import ActionResult
from nasim.envs.utils import get_minimal_hops_to_goal, min_subnet_depth, AccessLevel
from logger import logger

# column in topology adjacency matrix that represents connection between
# subnet and public
INTERNET = 0


class Network:
    """A computer network """

    def __init__(self, scenario):
        self.hosts = scenario.hosts
        self.host_num_map = scenario.host_num_map
        self.subnets = scenario.subnets
        self.topology = scenario.topology
        self.firewall = scenario.firewall
        self.address_space = scenario.address_space
        self.address_space_bounds = scenario.address_space_bounds
        self.sensitive_addresses = scenario.sensitive_addresses
        self.sensitive_hosts = scenario.sensitive_hosts

    def reset(self, state):
        """Reset the network state to initial state """
        next_state = state.copy()
        for host_addr in self.address_space:
            host = next_state.get_host(host_addr)
            host.compromised = False
            host.access = AccessLevel.NONE
            host.reachable = self.subnet_public(host_addr[0])
            host.discovered = host.reachable
        return next_state

    def perform_action(self, state, action, DEBUG=True):
        """Perform the given Action against the network.

        Arguments
        ---------
        state : State
            the current state
        action : Action
            the action to perform

        Returns
        -------
        State
            the state after the action is performed
        ActionObservation
            the result from the action
        """
        # Sử dụng biến DEBUG để hiển thị thông tin debug
        if DEBUG:
            logger.debug(f"[DEBUG] Bắt đầu perform_action với action: {action}")
        
        tgt_subnet, tgt_id = action.target
        # Kiểm tra tính hợp lệ của subnet và host ID
        # Đảm bảo subnet ID phải nằm trong phạm vi hợp lệ (0 < subnet < tổng số subnet)
        # Subnet 0 thường được dành riêng cho Internet/public network (INTERNET = 0)
        if DEBUG:
            logger.debug(f"[DEBUG] Kiểm tra tính hợp lệ: subnet={tgt_subnet}, host_id={tgt_id}")
            logger.debug(f"[DEBUG] Có {len(self.subnets)} subnet trong mạng")
        assert 0 < tgt_subnet < len(self.subnets)
        
        # Đảm bảo host ID không vượt quá số lượng host tối đa được định nghĩa cho subnet đó
        # self.subnets[tgt_subnet] chứa số lượng host tối đa trong subnet này
        # Host ID hợp lệ có giá trị từ 1 đến số lượng host tối đa
        assert tgt_id <= self.subnets[tgt_subnet]

        # Tạo bản sao của state hiện tại để không ảnh hưởng trực tiếp đến state gốc
        next_state = state.copy()
        if DEBUG:
            logger.debug(f"[DEBUG] Đã tạo bản sao của state hiện tại")

        # Nếu hành động là noop (không làm gì), trả về state mới và kết quả thành công
        if action.is_noop():
            if DEBUG:
                logger.debug(f"[DEBUG] Hành động noop - không làm gì")
            return next_state, ActionResult(True)

        # Kiểm tra xem host đích có thể truy cập và đã được phát hiện chưa
        host_reachable = state.host_reachable(action.target)
        host_discovered = state.host_discovered(action.target)
        if DEBUG:
            logger.debug(f"[DEBUG] Kiểm tra host đích: reachable={host_reachable}, discovered={host_discovered}")
        
        if not host_reachable or not host_discovered:
            # Nếu host đích không thể truy cập hoặc chưa được phát hiện
            # thì hành động sẽ thất bại với lỗi kết nối
            result = ActionResult(False, 0.0, connection_error=True)
            if DEBUG:
                logger.debug(f"[DEBUG] Lỗi kết nối: Host không thể truy cập hoặc chưa được phát hiện")
            return next_state, result

        # Kiểm tra quyền truy cập từ xa
        has_req_permission = self.has_required_remote_permission(state, action)
        if DEBUG:
            logger.debug(f"[DEBUG] Kiểm tra quyền truy cập từ xa: has_req_permission={has_req_permission}")
        
        if action.is_remote() and not has_req_permission:
            # Nếu hành động là từ xa và không có đủ quyền, thất bại với lỗi quyền
            result = ActionResult(False, 0.0, permission_error=True)
            if DEBUG:
                logger.debug(f"[DEBUG] Lỗi quyền: Không có quyền thực hiện hành động từ xa")
            return next_state, result

        # Kiểm tra liệu lưu lượng mạng đến dịch vụ đích có được phép không
        traffic_permitted = True
        if action.is_exploit():
            traffic_permitted = self.traffic_permitted(state, action.target, action.service)
            if DEBUG:
                logger.debug(f"[DEBUG] Kiểm tra lưu lượng mạng: permitted={traffic_permitted}, service={action.service}")
        
        if action.is_exploit() and not traffic_permitted:
            # Nếu là hành động khai thác nhưng lưu lượng bị chặn, thất bại với lỗi kết nối
            result = ActionResult(False, 0.0, connection_error=True)
            if DEBUG:
                logger.debug(f"[DEBUG] Lỗi kết nối: Lưu lượng mạng bị chặn bởi tường lửa")
            return next_state, result

        # Kiểm tra điều kiện leo thang đặc quyền
        host_compromised = state.host_compromised(action.target)
        if DEBUG:
            logger.debug(f"[DEBUG] Kiểm tra host đã xâm nhập: compromised={host_compromised}")
        
        if action.is_privilege_escalation() and not host_compromised:
            # Không thể leo thang đặc quyền trên host chưa bị xâm nhập
            result = ActionResult(False, 0.0, connection_error=True)
            if DEBUG:
                logger.debug(f"[DEBUG] Lỗi leo thang đặc quyền: Host chưa bị xâm nhập")
            return next_state, result
        
        if action.is_process_scan() and not host_compromised:
            # Không thể quét quy trình trên host chưa bị xâm nhập
            result = ActionResult(False, 0.0, connection_error=True)
            if DEBUG:
                logger.debug(f"[DEBUG] Lỗi quét quy trình: Host chưa bị xâm nhập")
            return next_state, result

        # Xử lý tính ngẫu nhiên trong thành công của hành động
        if action.is_exploit() and host_compromised:
            # Nếu host đã bị xâm nhập thì hành động khai thác luôn thành công
            # (bỏ qua yếu tố ngẫu nhiên)
            if DEBUG:
                logger.debug(f"[DEBUG] Host đã bị xâm nhập - hành động khai thác luôn thành công")
            pass
        elif np.random.rand() > action.prob:
            # Nếu số ngẫu nhiên lớn hơn xác suất thành công của hành động
            # thì hành động thất bại với lỗi không xác định
            if DEBUG:
                logger.debug(f"[DEBUG] Thất bại ngẫu nhiên: rand > action.prob ({action.prob})")
            return next_state, ActionResult(False, 0.0, undefined_error=True)

        # Xử lý trường hợp đặc biệt: quét subnet
        if action.is_subnet_scan():
            if DEBUG:
                logger.debug(f"[DEBUG] Thực hiện quét subnet")
            return self._perform_subnet_scan(next_state, action)

        # Thực hiện hành động trên host đích
        if DEBUG:
            logger.debug(f"[DEBUG] Thực hiện hành động trên host đích: {action.target}")
        
        t_host = state.get_host(action.target)
        next_host_state, action_obs = t_host.perform_action(action)
        
        # Cập nhật trạng thái host trong trạng thái mạng mới
        next_state.update_host(action.target, next_host_state)
        
        # Cập nhật các thông tin liên quan khác sau khi thực hiện hành động
        if DEBUG:
            logger.debug(f"[DEBUG] Cập nhật thông tin liên quan sau khi thực hiện hành động")
        self._update(next_state, action, action_obs)
        
        # Trả về trạng thái mới và kết quả quan sát được
        if DEBUG:
            logger.debug(f"[DEBUG] Kết thúc perform_action: success={action_obs.success}, reward={action_obs.value}")
        
        return next_state, action_obs

    def _perform_subnet_scan(self, next_state, action):
        # In thông tin đầu vào
        logger.debug(f"[DEBUG] Bắt đầu subnet_scan với target={action.target}, req_access={action.req_access}")
        
        # Kiểm tra máy chủ đã bị xâm nhập chưa
        if not next_state.host_compromised(action.target):
            logger.debug(f"[DEBUG] Thất bại: Host {action.target} chưa bị xâm nhập")
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        # Kiểm tra quyền truy cập
        if not next_state.host_has_access(action.target, action.req_access):
            logger.debug(f"[DEBUG] Thất bại: Host {action.target} không có đủ quyền truy cập. Yêu cầu: {action.req_access}")
            result = ActionResult(False, 0.0, permission_error=True)
            return next_state, result

        # Khởi tạo các biến theo dõi
        discovered = {}
        newly_discovered = {}
        discovery_reward = 0
        target_subnet = action.target[0]
        
        logger.debug(f"[DEBUG] Subnet mục tiêu: {target_subnet}")
        logger.debug(f"[DEBUG] Tổng số địa chỉ trong không gian địa chỉ: {len(self.address_space)}")

        # Lặp qua tất cả các host trong không gian địa chỉ
        for h_addr in self.address_space:
            newly_discovered[h_addr] = False
            discovered[h_addr] = False
            
            # Kiểm tra kết nối subnet
            subnet_connected = self.subnets_connected(target_subnet, h_addr[0])
            logger.debug(f"[DEBUG] Kiểm tra host {h_addr}: subnet_connected={subnet_connected}")
            
            if subnet_connected:
                host = next_state.get_host(h_addr)
                discovered[h_addr] = True
                
                # Kiểm tra xem host đã được phát hiện trước đó chưa
                if not host.discovered:
                    newly_discovered[h_addr] = True
                    host.discovered = True
                    discovery_reward += host.discovery_value
                    logger.debug(f"[DEBUG] Phát hiện mới host {h_addr}: discovery_value={host.discovery_value}")
                else:
                    logger.debug(f"[DEBUG] Host {h_addr} đã được phát hiện trước đó")

        # Tổng kết kết quả
        logger.debug(f"[DEBUG] Tổng số host đã phát hiện: {sum(1 for v in discovered.values() if v)}")
        logger.debug(f"[DEBUG] Tổng số host mới phát hiện: {sum(1 for v in newly_discovered.values() if v)}")
        logger.debug(f"[DEBUG] Tổng điểm thưởng khám phá: {discovery_reward}")

        # Tạo kết quả và trả về
        obs = ActionResult(
            True,
            discovery_reward,
            discovered=discovered,
            newly_discovered=newly_discovered
        )
        logger.debug(f"[DEBUG] Hoàn thành subnet_scan: success=True, reward={discovery_reward}")
        
        return next_state, obs

    def _update(self, state, action, action_obs):
        if action.is_exploit() and action_obs.success:
            self._update_reachable(state, action.target)

    def _update_reachable(self, state, compromised_addr):
        """Updates the reachable status of hosts on network, based on current
        state and newly exploited host
        """
        comp_subnet = compromised_addr[0]
        for addr in self.address_space:
            if state.host_reachable(addr):
                continue
            if self.subnets_connected(comp_subnet, addr[0]):
                state.set_host_reachable(addr)

    def get_sensitive_hosts(self):
        return self.sensitive_addresses

    def is_sensitive_host(self, host_address):
        return host_address in self.sensitive_addresses

    def subnets_connected(self, subnet_1, subnet_2):
        return self.topology[subnet_1][subnet_2] == 1

    def subnet_traffic_permitted(self, src_subnet, dest_subnet, service):
        """Kiểm tra xem lưu lượng mạng giữa hai subnet với dịch vụ cụ thể có được phép hay không.
        
        Phương thức này xác định liệu lưu lượng mạng từ subnet nguồn đến subnet đích 
        sử dụng dịch vụ cụ thể có được tường lửa cho phép hay không.
        
        Parameters
        ----------
        src_subnet : int
            Subnet nguồn của lưu lượng mạng
        dest_subnet : int
            Subnet đích của lưu lượng mạng
        service : int
            Dịch vụ được sử dụng cho lưu lượng mạng
            
        Returns
        -------
        bool
            True nếu lưu lượng mạng được phép, False nếu không
            
        Notes
        -----
        Lưu lượng mạng được phép nếu:
        - Subnet nguồn và đích là cùng một subnet
        - Các subnet có kết nối với nhau và dịch vụ được tường lửa cho phép
        """
        if src_subnet == dest_subnet:
            # in same subnet so permitted
            return True
        if not self.subnets_connected(src_subnet, dest_subnet):
            return False
        return service in self.firewall[(src_subnet, dest_subnet)]

    def host_traffic_permitted(self, src_addr, dest_addr, service):
        """Kiểm tra xem lưu lượng mạng từ host nguồn đến host đích có được phép hay không.
        
        Phương thức này kiểm tra xem host đích có cho phép lưu lượng mạng từ host nguồn
        sử dụng dịch vụ được chỉ định hay không.
        
        Args:
            src_addr (tuple): Địa chỉ của host nguồn
            dest_addr (tuple): Địa chỉ của host đích
            service (str): Tên dịch vụ cần kiểm tra
            
        Returns:
            bool: True nếu lưu lượng mạng được phép, False nếu không
        """
        dest_host = self.hosts[dest_addr]
        return dest_host.traffic_permitted(src_addr, service)

    def has_required_remote_permission(self, state, action, DEBUG = True):
        """Kiểm tra xem người tấn công có quyền cần thiết để thực hiện hành động từ xa không
        
        Parameters
        ----------
        state : State
            Trạng thái hiện tại của mạng
        action : Action
            Hành động cần kiểm tra quyền
            
        Returns
        -------
        bool
            True nếu có quyền thực hiện, False nếu không
        """
        # Thêm debug hiển thị bắt đầu kiểm tra với tên hàm
        if DEBUG:
            logger.debug(f"[DEBUG - has_required_remote_permission] Bắt đầu kiểm tra quyền truy cập từ xa: action={action}, target={action.target}")
        
        # Nếu subnet đích là public (có kết nối Internet), luôn cho phép thực hiện
        # vì các subnet public có thể được truy cập trực tiếp từ bên ngoài
        is_public = self.subnet_public(action.target[0])
        if DEBUG:
            logger.debug(f"[DEBUG - has_required_remote_permission] Kiểm tra subnet đích {action.target[0]} là public: {is_public}")
        
        if is_public:
            if DEBUG:
                logger.debug(f"[DEBUG - has_required_remote_permission] Subnet đích là public - cho phép thực hiện")
            return True
    
        # Thêm debug số lượng host trong không gian địa chỉ
        if DEBUG:
            logger.debug(f"[DEBUG - has_required_remote_permission] Tìm kiếm bàn đạp trong {len(self.address_space)} hosts")
        
        # Lặp qua tất cả các host trong mạng để tìm một host đã bị xâm nhập
        # có thể thực hiện hành động từ xa này
        for src_addr in self.address_space:
            # Debug thông tin host đang kiểm tra
            if DEBUG:
                host_compromised = state.host_compromised(src_addr)
                logger.debug(f"[DEBUG - has_required_remote_permission] Kiểm tra host {src_addr} làm bàn đạp: compromised={host_compromised}")
            
            # Bỏ qua các host chưa bị xâm nhập vì không thể dùng làm bàn đạp tấn công
            if not state.host_compromised(src_addr):
                continue
                
            # Kiểm tra kết nối subnet cho hành động scan  
            if action.is_scan():
                subnets_connected = self.subnets_connected(src_addr[0], action.target[0])
                if DEBUG:
                    logger.debug(f"[DEBUG - has_required_remote_permission] Kiểm tra kết nối giữa subnet {src_addr[0]} và {action.target[0]}: {subnets_connected}")
                
                if not subnets_connected:
                    continue
                    
            # Kiểm tra tường lửa cho hành động exploit
            if action.is_exploit():
                traffic_permitted = self.subnet_traffic_permitted(
                    src_addr[0], action.target[0], action.service
                )
                if DEBUG:
                    logger.debug(f"[DEBUG - has_required_remote_permission] Kiểm tra lưu lượng từ subnet {src_addr[0]} đến {action.target[0]} với service {action.service}: {traffic_permitted}")
                
                if not traffic_permitted:
                    continue
                    
            # Kiểm tra quyền truy cập của host
            has_access = state.host_has_access(src_addr, action.req_access)
            if DEBUG:
                logger.debug(f"[DEBUG - has_required_remote_permission] Kiểm tra host {src_addr} có quyền truy cập {action.req_access}: {has_access}")
                
            if has_access:
                if DEBUG:
                    logger.debug(f"[DEBUG - has_required_remote_permission] Tìm thấy bàn đạp phù hợp: {src_addr}")
                return True
                    
        # Không tìm thấy host bàn đạp nào thỏa mãn điều kiện
        if DEBUG:
            logger.debug(f"[DEBUG - has_required_remote_permission] Không tìm thấy host bàn đạp nào phù hợp - từ chối thực hiện")
        
        # Nếu không tìm thấy host nào thỏa mãn các điều kiện trên
        return False

    def traffic_permitted(self, state, host_addr, service):
        """Kiểm tra xem tường lửa của subnet và host có cho phép lưu lượng truy cập đến
        một host và dịch vụ cụ thể hay không, dựa trên tập hợp các host đã bị xâm phạm
        trong mạng.

        Parameters:
        -----------
        state : State
            Trạng thái hiện tại của mạng, chứa thông tin về các host đã bị xâm phạm.
        host_addr : tuple
            Địa chỉ của host đích mà lưu lượng truy cập hướng đến, dưới dạng (subnet_id, host_id).
        service : int or str
            Dịch vụ mà lưu lượng truy cập hướng đến.

        Returns:
        --------
        bool
            True nếu lưu lượng truy cập được phép (có ít nhất một host đã bị xâm phạm
            có thể truy cập vào host_addr và service), False nếu không.
        
        Notes:
        ------
        Hàm này kiểm tra từng host đã bị xâm phạm trong mạng để xem liệu có host nào
        có thể gửi lưu lượng truy cập đến host đích và dịch vụ cụ thể không. Nếu có
        ít nhất một host như vậy, hàm trả về True.
        """
        for src_addr in self.address_space:
            # host nguồn phải đã bị xâm phạm HOẶC nằm trong subnet công khai
            if not state.host_compromised(src_addr) and \
               not self.subnet_public(src_addr[0]):
                continue
            # kiểm tra xem tường lửa của subnet có cho phép lưu lượng từ subnet nguồn đến subnet đích và dịch vụ không
            if not self.subnet_traffic_permitted(
                    src_addr[0], host_addr[0], service
            ):
                continue
            # kiểm tra tường lửa cụ thể của host đích xem có cho phép lưu lượng từ host nguồn đến dịch vụ cụ thể không
            if self.host_traffic_permitted(src_addr, host_addr, service):
                return True
        return False

    def subnet_public(self, subnet):
        """
        Kiểm tra xem subnet có phải là public hay không. (có kết nối Internet)
        
        Một subnet được coi là public nếu nó có kết nối trực tiếp với Internet.
        
        Parameters
        ----------
        subnet : int
            Chỉ số (ID) của subnet cần kiểm tra
            
        Returns
        -------
        bool
            True nếu subnet là public (có kết nối Internet), False nếu ngược lại
        """
        return self.topology[subnet][INTERNET] == 1

    def get_number_of_subnets(self):
        return len(self.subnets)

    def all_sensitive_hosts_compromised(self, state):
        """Kiểm tra xem tất cả các host nhạy cảm có bị xâm nhập hay không.

        Duyệt qua tất cả các địa chỉ host nhạy cảm và kiểm tra xem trạng thái hiện tại
        có quyền truy cập ROOT vào host đó hay không.

        Args:
            state: Trạng thái hiện tại của môi trường.

        Returns:
            bool: True nếu tất cả các host nhạy cảm đã bị xâm nhập với quyền ROOT,
                  False nếu có ít nhất một host nhạy cảm chưa bị xâm nhập.
        """
        for host_addr in self.sensitive_addresses:
            if not state.host_has_access(host_addr, AccessLevel.ROOT):
                return False
        return True

    def get_total_sensitive_host_value(self):
        """Tính tổng giá trị của tất cả các host nhạy cảm trong mạng.

        Returns:
            float: Tổng giá trị của tất cả các host nhạy cảm

        Lưu ý:
            Hàm này duyệt qua tất cả các host nhạy cảm và cộng dồn giá trị của chúng.
            Kết quả được sử dụng để tính toán phần thưởng trong môi trường.
        """
        total = 0
        for host_value in self.sensitive_hosts.values():
            total += host_value
        return total

    def get_total_discovery_value(self):
        """Tính tổng giá trị khám phá (discovery value) của tất cả các host trong mạng.
        
        Phương thức này duyệt qua từng host trong mạng và cộng dồn giá trị khám phá của chúng.
        
        Returns:
            int: Tổng giá trị khám phá của tất cả các host trong mạng.
        """
        total = 0
        for host in self.hosts.values():
            total += host.discovery_value
        return total

    def get_minimal_hops(self):
        """Lấy số bước tối thiểu để đến các host nhạy cảm.
        
        Phương thức này tính toán số bước nhảy (hop) tối thiểu từ bất kỳ host 
        khởi đầu nào để đạt được các host nhạy cảm trong mạng.
        
        Returns:
            dict: Từ điển ánh xạ từ địa chỉ host đến số bước nhảy tối thiểu cần thiết 
                  để đạt đến các host nhạy cảm từ host đó.
        
        Note:
            Sử dụng hàm get_minimal_hops_to_goal để thực hiện tính toán dựa trên 
            cấu trúc liên kết mạng và danh sách các host nhạy cảm.
        """
        # Trả về số bước nhảy tối thiểu để đến các host nhạy cảm từ mỗi host trong mạng
        return get_minimal_hops_to_goal(
            self.topology, self.sensitive_addresses
        )

    def get_subnet_depths(self):
        """
        Lấy chiều sâu tối thiểu của các subnet trong mạng.
        
        Returns:
            dict: Từ điển chứa chiều sâu tối thiểu của các subnet, 
                  với khóa là subnet và giá trị là chiều sâu.
                  
        Note:
            Chiều sâu của subnet được tính là số bước nhảy tối thiểu 
            từ internet đến subnet đó.
        """
        return min_subnet_depth(self.topology)

    def __str__(self):
        output = "\n--- Network ---\n"
        output += "Subnets: " + str(self.subnets) + "\n"
        output += "Topology:\n"
        for row in self.topology:
            output += f"\t{row}\n"
        output += "Sensitive hosts: \n"
        for addr, value in self.sensitive_hosts.items():
            output += f"\t{addr}: {value}\n"
        output += "Num_services: {self.scenario.num_services}\n"
        output += "Hosts:\n"
        for m in self.hosts.values():
            output += str(m) + "\n"
        output += "Firewall:\n"
        for c, a in self.firewall.items():
            output += f"\t{c}: {a}\n"
        return output
