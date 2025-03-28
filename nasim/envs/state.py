import numpy as np

from nasim.envs.host_vector import HostVector
from nasim.envs.observation import Observation


class State:
    """Một trạng thái trong NASim Environment.

    Mỗi hàng trong tensor trạng thái đại diện cho trạng thái của một máy chủ 
    trong mạng. Để biết chi tiết về cách biểu diễn trạng thái của một máy chủ, 
    hãy xem :class:`HostVector`

    ...

    Thuộc tính
    ----------
    tensor : numpy.Array
        biểu diễn tensor của trạng thái mạng
    host_num_map : dict
        ánh xạ từ địa chỉ máy chủ đến số máy chủ (index) (được sử dụng
        để ánh xạ địa chỉ máy chủ đến hàng tương ứng trong tensor mạng)
    """

    def __init__(self, network_tensor, host_num_map):
        """
        Thuộc tính
        ----------
        tensor : numpy.Array
            biểu diễn tensor của trạng thái mạng
        host_num_map : dict
            ánh xạ từ địa chỉ máy chủ đến số máy chủ (index) (được sử dụng
            để ánh xạ địa chỉ máy chủ đến hàng tương ứng trong tensor mạng)
        """
        self.tensor = network_tensor
        self.host_num_map = host_num_map

    @classmethod
    def tensorize(cls, network):
        """Chuyển đổi mạng thành tensor để biểu diễn trạng thái.

        Phương thức này tạo ra một tensor chứa thông tin vector hóa của tất cả các host
        trong mạng. Mỗi hàng của tensor tương ứng với một host, và số cột bằng với kích thước
        vector trạng thái của host.

        Args:
            network: Đối tượng mạng chứa thông tin về các host và không gian địa chỉ

        Returns:
            Một đối tượng StateVector chứa tensor biểu diễn trạng thái và ánh xạ số host

        Chi tiết:
            - Đầu tiên, lấy host đầu tiên (1,0) để xác định kích thước vector
            - Tạo tensor rỗng với kích thước phù hợp
            - Điền thông tin vector hóa của từng host vào tensor
            - Trả về đối tượng StateVector với tensor và ánh xạ số host
        """
        h0 = network.hosts[(1, 0)]  # Lấy host đầu tiên từ network
        h0_vector = HostVector.vectorize(h0, network.address_space_bounds)  # Chuyển đổi host đầu tiên thành vector
        tensor = np.zeros(
            (len(network.hosts), h0_vector.state_size),  # Tạo tensor với kích thước (số lượng host, kích thước vector của mỗi host)
            dtype=np.float32
        )
        for host_addr, host in network.hosts.items():  # Duyệt qua từng host trong network
            host_num = network.host_num_map[host_addr]  # Lấy số thứ tự (index) của host từ map
            HostVector.vectorize(
                host, network.address_space_bounds, tensor[host_num]  # Chuyển đổi host thành vector và lưu vào tensor
            )
        return cls(tensor, network.host_num_map)  # Trả về đối tượng State mới với tensor và host_num_map

    @classmethod
    def generate_initial_state(cls, network):
        cls.reset()
        state = cls.tensorize(network)
        return network.reset(state)

    @classmethod
    def generate_random_initial_state(cls, network):
        h0 = network.hosts[(1, 0)]
        h0_vector = HostVector.vectorize_random(
            h0, network.address_space_bounds
        )
        tensor = np.zeros(
            (len(network.hosts), h0_vector.state_size),
            dtype=np.float32
        )
        for host_addr, host in network.hosts.items():
            host_num = network.host_num_map[host_addr]
            HostVector.vectorize_random(
                host, network.address_space_bounds, tensor[host_num]
            )
        state = cls(tensor, network.host_num_map)
        # ensure host state set correctly
        return network.reset(state)

    @classmethod
    def from_numpy(cls, s_array, state_shape, host_num_map):
        if s_array.shape != state_shape:
            s_array = s_array.reshape(state_shape)
        return State(s_array, host_num_map)

    @classmethod
    def reset(cls):
        """Reset any class attributes for state """
        HostVector.reset()

    @property
    def hosts(self):
        hosts = []
        for host_addr in self.host_num_map:
            hosts.append((host_addr, self.get_host(host_addr)))
        return hosts

    def copy(self):
        new_tensor = np.copy(self.tensor)
        return State(new_tensor, self.host_num_map)

    def get_initial_observation(self, fully_obs):
        """Lấy quan sát ban đầu của mạng.
    
        Parameters
        ----------
        fully_obs : bool
            Chế độ quan sát của môi trường (True: quan sát đầy đủ, False: quan sát một phần)
    
        Returns
        -------
        Observation
            Đối tượng observation chứa thông tin quan sát ban đầu
        """
        # Tạo đối tượng Observation trống với kích thước bằng với trạng thái hiện tại
        obs = Observation(self.shape())
        
        # Nếu là chế độ quan sát đầy đủ (fully observable)
        if fully_obs:
            # Điền toàn bộ thông tin từ trạng thái hiện tại vào observation
            # Agent sẽ biết được tất cả thông tin về mạng, bao gồm các máy chưa khám phá
            obs.from_state(self)
            return obs
    
        # Nếu là chế độ quan sát một phần (partially observable)
        # Chỉ điền thông tin của các máy chủ có thể tiếp cận (reachable)
        for host_addr, host in self.hosts:
            # Bỏ qua các máy chủ không thể tiếp cận
            if not host.reachable:
                continue
            
            # Chỉ lấy thông tin cơ bản: địa chỉ, khả năng tiếp cận, và trạng thái đã khám phá
            # Agent chỉ biết về các máy chủ có sẵn ban đầu, thông thường chỉ là máy chủ bắt đầu
            host_obs = host.observe(address=True,
                                    reachable=True,
                                    discovered=True)
            
            # Lấy chỉ mục của host trong tensor và cập nhật thông tin vào observation
            host_idx = self.get_host_idx(host_addr)
            obs.update_from_host(host_idx, host_obs)
        
        return obs

    def get_observation(self, action, action_result, fully_obs):
        """Get observation given last action and action result

        Parameters
        ----------
        action : Action
            last action performed
        action_result : ActionResult
            observation from performing action
        fully_obs : bool
            whether problem is fully observable or not

        Returns
        -------
        Observation
            an observation object
        """
        obs = Observation(self.shape())
        obs.from_action_result(action_result)
        if fully_obs:
            obs.from_state(self)
            return obs

        if action.is_noop():
            return obs

        if not action_result.success:
            # action failed so no observation
            return obs

        t_idx, t_host = self.get_host_and_idx(action.target)
        obs_kwargs = dict(
            address=True,       # must be true for success
            compromised=False,
            reachable=True,     # must be true for success
            discovered=True,    # must be true for success
            value=False,
            # discovery_value=False,    # this is only added as needed
            services=False,
            processes=False,
            os=False,
            access=False
        )
        if action.is_exploit():
            # exploit action, so get all observations for host
            obs_kwargs["compromised"] = True
            obs_kwargs["services"] = True
            obs_kwargs["os"] = True
            obs_kwargs["access"] = True
            obs_kwargs["value"] = True
        elif action.is_privilege_escalation():
            obs_kwargs["compromised"] = True
            obs_kwargs["access"] = True
        elif action.is_service_scan():
            obs_kwargs["services"] = True
        elif action.is_os_scan():
            obs_kwargs["os"] = True
        elif action.is_process_scan():
            obs_kwargs["processes"] = True
            obs_kwargs["access"] = True
        elif action.is_subnet_scan():
            for host_addr in action_result.discovered:
                discovered = action_result.discovered[host_addr]
                if not discovered:
                    continue
                d_idx, d_host = self.get_host_and_idx(host_addr)
                newly_discovered = action_result.newly_discovered[host_addr]
                d_obs = d_host.observe(
                    discovery_value=newly_discovered, **obs_kwargs
                )
                obs.update_from_host(d_idx, d_obs)
            # this is for target host (where scan was performed on)
            obs_kwargs["compromised"] = True
        else:
            raise NotImplementedError(f"Action {action} not implemented")
        target_obs = t_host.observe(**obs_kwargs)
        obs.update_from_host(t_idx, target_obs)
        return obs

    def shape_flat(self):
        return self.numpy_flat().shape

    def shape(self):
        return self.tensor.shape

    def numpy_flat(self):
        return self.tensor.flatten()

    def numpy(self):
        return self.tensor

    def update_host(self, host_addr, host_vector):
        host_idx = self.host_num_map[host_addr]
        self.tensor[host_idx] = host_vector.vector

    def get_host(self, host_addr):
        host_idx = self.host_num_map[host_addr]
        return HostVector(self.tensor[host_idx])

    def get_host_idx(self, host_addr):
        return self.host_num_map[host_addr]

    def get_host_and_idx(self, host_addr):
        host_idx = self.host_num_map[host_addr]
        return host_idx, HostVector(self.tensor[host_idx])

    def host_reachable(self, host_addr):
        return self.get_host(host_addr).reachable

    def host_compromised(self, host_addr):
        return self.get_host(host_addr).compromised

    def host_discovered(self, host_addr):
        return self.get_host(host_addr).discovered

    def host_has_access(self, host_addr, access_level):
        return self.get_host(host_addr).access >= access_level

    def set_host_compromised(self, host_addr):
        self.get_host(host_addr).compromised = True

    def set_host_reachable(self, host_addr):
        self.get_host(host_addr).reachable = True

    def set_host_discovered(self, host_addr):
        self.get_host(host_addr).discovered = True

    def get_host_value(self, host_address):
        return self.hosts[host_address].get_value()

    def host_is_running_service(self, host_addr, service):
        return self.get_host(host_addr).is_running_service(service)

    def host_is_running_os(self, host_addr, os):
        return self.get_host(host_addr).is_running_os(os)

    def get_total_host_value(self):
        total_value = 0
        for host_addr in self.host_num_map:
            host = self.get_host(host_addr)
            total_value += host.value
        return total_value

    def state_size(self):
        return self.tensor.size

    def get_readable(self):
        host_obs = []
        for host_addr in self.host_num_map:
            host = self.get_host(host_addr)
            readable_dict = host.readable()
            host_obs.append(readable_dict)
        return host_obs

    def __str__(self):
        output = "\n--- State ---\n"
        output += "Hosts:\n"
        for host in self.hosts:
            output += str(host) + "\n"
        return output

    def __hash__(self):
        return hash(str(self.tensor))

    def __eq__(self, other):
        return np.array_equal(self.tensor, other.tensor)
