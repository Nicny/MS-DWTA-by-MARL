from mozi_ai_sdk.base_env import BaseEnvironment
from mozi_utils.geo import get_two_point_distance, get_degree
import numpy as np
import math
import os


class AirDefense(BaseEnvironment):
    def __init__(self, server_ip=None, server_port=None, agent_key_event_file=None, duration_interval=None,
                 app_mode=None, synchronous=None, simulate_compression=None, scenario_name=None,
                 platform_mode=None, platform="windows", args=None):
        super().__init__(server_ip, server_port, platform, scenario_name, simulate_compression,
                         duration_interval, synchronous, app_mode, platform_mode)

        self.args = args
        self.SERVER_PLAT = platform
        self.agent_key_event_file = agent_key_event_file

        self.PI = 3.1415926535897932
        self.degree2radian = self.PI / 180.0

        self.red_side_name = "红方"
        self.blue_side_name = "蓝方"

        self.blue_ship = {}
        self.blue_missile = {}
        self.red_missile = {}
        self.rank_dic = {}

        self.n_agents = args.n_agents
        self.n_enemies = args.enemies
        self.n_actions = self.n_enemies * 4 + 1
        self.step_limit = args.episode_limit

        self.death_tracker_ally = [0 for i in range(self.n_agents)]
        self.prev_health = [1 for i in range(self.n_agents)]
        self.now_health = [1 for i in range(self.n_agents)]
        self.health_loss = [0 for i in range(self.step_limit)]
        self.prev_mount = [[61, 32, 32, 21] for i in range(self.n_agents - 1)] + [[21, 21, 21, 21]]

        self.ship_value = args.ship_value
        self.all_avail_target = [0] * self.n_enemies

        self.step = 0
        self.fire_range_1 = 92600
        self.min_fire_range_1 = 3704
        self.fire_range_2 = 240760
        self.min_fire_range_2 = 7408
        self.fire_range_3 = 92600
        self.min_fire_range_3 = 3704
        self.fire_range_4 = 25520
        self.min_fire_range_4 = 371

        self.border = 30000
        self.own_defense = 10000

        self.weapon_guid = ['hsfw-dataweapon-00000000001194', 'hsfw-dataweapon-00000000001310',
                            'hsfw-dataweapon-00000000001195', 'hsfw-dataweapon-00000000002853', ]
        self.weapon_reward = [-0.03, -0.05, -0.02, -0.04]

        self.fire_facility = ["a1", "a2", "a3", "a4", "a5",
                              "b1", "b2", "b3", "b4", "b5",
                              "c1", "c2", "c3", "c4", "c5",
                              "d1", "d2", "d3", "d4", "d5"]

        self.dr = True        # 动态奖励法调整
        self.all_impact_dist = 0
        self.target = []
        print("环境验证：是否动态奖励：", self.dr)

    def reset(self, app_mode=None):
        super(AirDefense, self).reset()
        self.blue_ship = {}
        self.blue_missile = {}
        self.red_missile = {}
        self.rank_dic = {}

        self.death_tracker_ally = [0 for i in range(self.n_agents)]
        self.prev_health = [1 for i in range(self.n_agents)]
        self.now_health = [1 for i in range(self.n_agents)]
        self.health_loss = [0 for i in range(self.step_limit)]
        self.prev_mount = [[61, 32, 32, 21] for i in range(self.n_agents - 1)] + [[21, 21, 21, 21]]

        self.all_avail_target = [0] * self.n_enemies
        self.step = 0
        self.all_impact_dist = 0

        self._construct_side_entity()
        self._init_unit_list()
        self.scenario = super(AirDefense, self).step()

    def _construct_side_entity(self):
        self.redside = self.scenario.get_side_by_name(self.red_side_name)
        self.redside.static_construct()
        self.blueside = self.scenario.get_side_by_name(self.blue_side_name)
        self.blueside.static_construct()

    def _init_unit_list(self):
        temp_value_1 = {}
        for key, value in self.blueside.ships.items():
            temp_value_1["guid"] = key
            temp_value_1["unit"] = value
            v = temp_value_1.copy()
            if self.blueside.ships[key].strName == "a0":
                blue_idx = 0
            elif self.blueside.ships[key].strName == "a1":
                blue_idx = 1
            elif self.blueside.ships[key].strName == "d0":
                blue_idx = 2
            elif self.blueside.ships[key].strName == "b0":
                blue_idx = 3
            elif self.blueside.ships[key].strName == "b1":
                blue_idx = 4
            elif self.blueside.ships[key].strName == "d1":
                blue_idx = 5
            else:
                blue_idx = 6
                self.center_lon = self.blueside.ships[key].dLongitude
                self.center_lat = self.blueside.ships[key].dLatitude
            self.blue_ship[blue_idx] = v

        temp_value_2 = {}
        for red_idx in range(self.n_enemies):
            temp_value_2["guid"] = None
            temp_value_2["unit"] = None
            self.red_missile[red_idx] = temp_value_2.copy()

        temp_value_3 = {}
        for idx in range(len(self.fire_facility)):
            temp_value_3["fire_unit"] = self.fire_facility[idx]
            temp_value_3["name"] = []
            self.rank_dic[idx] = temp_value_3.copy()

    def get_health_agent(self, agent_id):
        if self.death_tracker_ally[agent_id]:
            health = 0
        else:
            ship = self.blue_ship[agent_id]["unit"]
            damage = ship.strDamageState
            health = (100 - float(damage)) / 100
        return health

    def get_weapon_num(self, agent_id):
        weapon_num_list = []
        health = self.get_health_agent(agent_id)
        if health != 0:
            ship = self.blue_ship[agent_id]["unit"]
            total_weapon = ship.get_mounts()
            for key in total_weapon:
                weapon_status = total_weapon[key].m_ComponentStatus
                if weapon_status == 0:
                    weapon_num = total_weapon[key].strLoadWeaponCount
                    index = weapon_num.find('/')
                    weapon_num = int(weapon_num[1:index])
                else:
                    weapon_num = 0
                weapon_num_list.append(weapon_num)
        else:
            weapon_num_list = [0, 0, 0, 0]
        return weapon_num_list

    def get_avail_agent_actions(self, agent_id, actions):
        avail_actions = [1] + [0] * (self.n_actions - 1)
        health = self.get_health_agent(agent_id)
        target_chosen = [None] * self.n_agents
        if health > 0:
            ship = self.blue_ship[agent_id]["unit"]
            lon = ship.dLongitude
            lat = ship.dLatitude
            for i, action in enumerate(actions):
                if action > 0:
                    weapon = math.floor((action - 1) / self.n_enemies)
                    target_chosen[i] = action - weapon * self.n_enemies - 1

            for e_id, e_value in enumerate(self.all_avail_target):
                if e_value == 1 and e_id not in target_chosen:
                    e_unit = self.red_missile[e_id]["unit"]
                    e_lon = e_unit.dLongitude
                    e_lat = e_unit.dLatitude
                    own_dist = get_two_point_distance(lon, lat, e_lon, e_lat)
                    vessel_dist = get_two_point_distance(self.center_lon, self.center_lat, e_lon, e_lat)
                    weapon_permission = self.weapon_permit(agent_id, own_dist, vessel_dist)
                    for i, j in enumerate(weapon_permission):
                        if j == 1:
                            idx = e_id + self.n_enemies * i + 1
                            avail_actions[idx] = 1
        return avail_actions

    def weapon_permit(self, agent_id, own_dist, vessel_dist):
        weapon_permission = [0, 0, 0, 0]
        weapon_num_list = self.get_weapon_num(agent_id)
        if vessel_dist > self.border:
            if agent_id in [3, 4, 5, 6]:
                return weapon_permission
            else:
                if self.min_fire_range_1 < own_dist < self.fire_range_1 and weapon_num_list[0] > 0:
                    weapon_permission[0] = 1
                if self.min_fire_range_2 < own_dist < self.fire_range_2 and weapon_num_list[1] > 0:
                    weapon_permission[1] = 1
                if self.min_fire_range_3 < own_dist < self.fire_range_3 and weapon_num_list[2] > 0:
                    weapon_permission[2] = 1
                if self.min_fire_range_4 < own_dist < self.fire_range_4 and weapon_num_list[3] > 0:
                    weapon_permission[3] = 1
                return weapon_permission
        elif own_dist > self.own_defense:
            if agent_id in [0, 1, 2, 6]:
                return weapon_permission
            else:
                if self.min_fire_range_1 < own_dist < self.fire_range_1 and weapon_num_list[0] > 0:
                    weapon_permission[0] = 1
                if self.min_fire_range_2 < own_dist < self.fire_range_2 and weapon_num_list[1] > 0:
                    weapon_permission[1] = 1
                if self.min_fire_range_3 < own_dist < self.fire_range_3 and weapon_num_list[2] > 0:
                    weapon_permission[2] = 1
                if self.min_fire_range_4 < own_dist < self.fire_range_4 and weapon_num_list[3] > 0:
                    weapon_permission[3] = 1
                return weapon_permission
        else:
            if weapon_num_list[0] > 0:
                weapon_permission[0] = 1
            if weapon_num_list[1] > 0:
                weapon_permission[1] = 1
            if weapon_num_list[2] > 0:
                weapon_permission[2] = 1
            if weapon_num_list[3] > 0:
                weapon_permission[3] = 1
            return weapon_permission

    def get_dist_and_permission(self):
        dist_array = np.ones([self.n_agents, self.n_enemies]) * 99999999
        weapon_permission_array = [[[0, 0, 0, 0] for i in range(self.n_enemies)] for j in range(self.n_agents)]
        for e_id, e_value in enumerate(self.all_avail_target):
            if e_value == 1:
                e_unit = self.red_missile[e_id]["unit"]
                e_lon = e_unit.dLongitude
                e_lat = e_unit.dLatitude
                for al_id in self.blue_ship:
                    al_health = self.get_health_agent(al_id)
                    al_unit = self.blue_ship[al_id]["unit"]
                    if al_health > 0:
                        al_lon = al_unit.dLongitude
                        al_lat = al_unit.dLatitude
                        own_dist = get_two_point_distance(al_lon, al_lat, e_lon, e_lat)
                        vessel_dist = get_two_point_distance(self.center_lon, self.center_lat, e_lon, e_lat)
                        weapon_permission = self.weapon_permit(al_id, own_dist, vessel_dist)
                        if weapon_permission != [0, 0, 0, 0]:
                            dist_array[al_id, e_id] = own_dist
                            weapon_permission_array[al_id][e_id] = weapon_permission
        return dist_array, weapon_permission_array

    def get_idx(self, unit):
        name = unit.strName[-2:]
        for key, value in self.rank_dic.items():
            if name in value["name"]:
                index = value["name"].index(name)
                idx = int(key) * 2 + int(index)
                return idx
        else:
            print("=====================No index！========================")
            return None

    def record_name(self, unit):
        name = unit.strName[-2:]
        fire_unit_guid = unit.m_FiringUnitGuid
        fire_unit = self.scenario.situation.get_obj_by_guid(fire_unit_guid)
        if fire_unit:
            fire_unit_name = fire_unit.strName
            for key, value in self.rank_dic.items():
                if fire_unit_name == value["fire_unit"]:
                    if len(value["name"]) == 0:
                        value["name"].append(name)
                    elif len(value["name"]) == 1 and name not in value["name"]:
                        value["name"].append(name)

    def get_missile_list(self):
        for k, v in self.redside.weapons.items():
            if v.m_WeaponType == 2001:
                self.record_name(v)
        type_list = []
        self.all_avail_target = [0] * self.n_enemies
        for key in self.blueside.contacts:
            contact_type = self.blueside.contacts[key].m_ContactType
            type_list.append(int(contact_type))
            if contact_type == 1:
                unit = self.blueside.contacts[key]
                true_unit = unit.get_actual_unit()
                if true_unit and unit.iWeaponsAimingAtMe == 0:
                    idx = self.get_idx(true_unit)
                    # print("索引号为：", idx)
                    if idx is not None:
                        self.all_avail_target[idx] = 1
                        # print("索引号列表为：", self.all_avail_target)
                        if self.red_missile[idx]["guid"] is None:
                            self.red_missile[idx]["guid"] = key
                            self.red_missile[idx]["unit"] = true_unit
        # return self.all_avail_target, self.rank_dic

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        ship = self.blue_ship[agent_id]["unit"]
        enemy_feats = np.zeros((self.n_enemies, 6), dtype=np.float32)
        ally_feats = np.zeros((self.n_agents - 1, 8), dtype=np.float32)
        own_feats = np.zeros(7, dtype=np.float32)

        health = self.get_health_agent(agent_id)
        if health > 0:
            lon = ship.dLongitude
            lat = ship.dLatitude

            # avail_actions = self.get_avail_agent_actions(agent_id)
            for e_id, e_value in enumerate(self.all_avail_target):
                if e_value == 1:
                    e_unit = self.red_missile[e_id]["unit"]
                    e_speed = e_unit.fCurrentSpeed
                    e_name = e_unit.strName[:3]
                    # print("弹的名称为：", e_name)
                    if e_name == '海军打':
                        e_threat = 1
                    else:
                        e_threat = 0.8
                    e_lon = e_unit.dLongitude
                    e_lat = e_unit.dLatitude
                    dist = get_two_point_distance(lon, lat, e_lon, e_lat)

                    e_heading = e_unit.fCurrentHeading
                    angle = get_degree(lat, lon, e_lat, e_lon)
                    e_angle = abs(e_heading - angle)
                    if e_angle > 180:
                        e_angle -= 180
                    if e_angle <= 90:
                        e_angle = 1 - e_angle / 180
                    else:
                        e_angle = 1 - (180 - e_angle) / 180

                    enemy_feats[e_id, 0] = e_angle
                    enemy_feats[e_id, 1] = dist / 100000
                    enemy_feats[e_id, 2] = (e_lon - 100) / 100
                    enemy_feats[e_id, 3] = e_lat / 100
                    enemy_feats[e_id, 4] = e_speed / 1000
                    enemy_feats[e_id, 5] = e_threat

            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.blue_ship[al_id]["unit"]
                al_lon = al_unit.dLongitude
                al_lat = al_unit.dLatitude
                dist = get_two_point_distance(lon, lat, al_lon, al_lat)

                al_health = self.get_health_agent(al_id)
                if al_health > 0:
                    ally_feats[i, 0] = dist / 100000
                    ally_feats[i, 1] = (al_lon - 100) / 100
                    ally_feats[i, 2] = al_lat / 100
                    ally_feats[i, 3] = al_health

                    al_weapon = self.get_weapon_num(al_id)
                    ally_feats[i, 4] = al_weapon[0] / 100
                    ally_feats[i, 5] = al_weapon[1] / 100
                    ally_feats[i, 6] = al_weapon[2] / 100
                    ally_feats[i, 7] = al_weapon[3] / 100

            own_feats[0] = (lon - 100) / 100
            own_feats[1] = lat / 100
            own_feats[2] = health

            own_weapon = self.get_weapon_num(agent_id)
            own_feats[3] = own_weapon[0] / 100
            own_feats[4] = own_weapon[1] / 100
            own_feats[5] = own_weapon[2] / 100
            own_feats[6] = own_weapon[3] / 100

        agent_obs = np.concatenate(
            (
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten()
            )
        )
        return agent_obs

    def get_state(self):
        state_dict = self.get_state_dict()
        state = np.append(
            state_dict["allies"].flatten(), state_dict["enemies"].flatten()
        )
        state = state.astype(dtype=np.float32)
        return state

    def get_state_dict(self):
        num_feature_al = 7
        num_feature_en = 5

        ally_state = np.zeros((self.n_agents, num_feature_al))
        enemy_state = np.zeros((self.n_enemies, num_feature_en))

        for al_id in self.blue_ship:
            al_health = self.get_health_agent(al_id)
            al_unit = self.blue_ship[al_id]["unit"]
            if al_health > 0:
                ally_state[al_id, 0] = al_health
                ally_state[al_id, 1] = (al_unit.dLongitude - 100) / 100
                ally_state[al_id, 2] = al_unit.dLatitude / 100

                ally_weapon = self.get_weapon_num(al_id)
                ally_state[al_id, 3] = ally_weapon[0] / 100
                ally_state[al_id, 4] = ally_weapon[1] / 100
                ally_state[al_id, 5] = ally_weapon[2] / 100
                ally_state[al_id, 6] = ally_weapon[3] / 100

        for e_id, e_value in enumerate(self.all_avail_target):
            if e_value == 1:
                e_unit = self.red_missile[e_id]["unit"]
                e_speed = e_unit.fCurrentSpeed
                e_name = e_unit.strName
                e_name = e_name[:3]
                if e_name == '海军打':
                    e_threat = 1
                else:
                    e_threat = 0.8
                e_lon = e_unit.dLongitude
                e_lat = e_unit.dLatitude
                vessel_dist = get_two_point_distance(self.center_lon, self.center_lat, e_lon, e_lat)

                enemy_state[e_id, 0] = (e_lon - 100) / 100
                enemy_state[e_id, 1] = e_lat / 100
                enemy_state[e_id, 2] = vessel_dist / 100000
                enemy_state[e_id, 3] = e_speed / 1000
                enemy_state[e_id, 4] = e_threat

        state = {"allies": ally_state, "enemies": enemy_state}

        return state

    def scenario_run(self):
        super(AirDefense, self).step()
        # print("=================================================态势运行中")
        # terminal = False
        # return terminal

    def do_action(self, actions):
        step_use_mount = [0, 0, 0, 0]
        actions_int = [int(a) for a in actions]
        for a_id, action in enumerate(actions_int):
            self.prev_health[a_id] = self.get_health_agent(a_id)
            self.prev_mount[a_id] = self.get_weapon_num(a_id)
            unit = self.blue_ship[a_id]["unit"]
            if action > 0:
                weapon = math.floor((action - 1) / self.n_enemies)
                target = action - weapon * self.n_enemies - 1
                if a_id == 4:
                    weapon = 3
                target_guid = self.red_missile[target]["guid"]
                response = unit.allocate_weapon_to_target(target_guid, self.weapon_guid[weapon], 1)
                if response == 'lua执行成功':
                    step_use_mount[weapon] += 1
                else:
                    error_ship = unit.strName

        self.scenario = super(AirDefense, self).step()
        for k, v in self.scenario.situation.wpnimpact_dic.items():
            impact_lon = v.dLongitude
            impact_lat = v.dLatitude
            impact_dist = get_two_point_distance(self.center_lon, self.center_lat, impact_lon, impact_lat)
            self.all_impact_dist += impact_dist

        for a_id in range(self.n_agents):
            self.now_health[a_id] = self.get_health_agent(a_id)
            health_loss = self.ship_value[a_id] * (self.now_health[a_id] - self.prev_health[a_id])
            if health_loss <= -0.05:
                self.health_loss[self.step] += health_loss
            else:
                pass

        self.step += 1
        return step_use_mount

    def reward_battle(self, mount_use_ep):
        r = []
        mount_use_sum = [0, 0, 0, 0]

        for mount_use_step in mount_use_ep:
            step_reward = 0
            for weapon_id in range(len(self.weapon_reward)):
                step_reward += mount_use_step[weapon_id] * self.weapon_reward[weapon_id]
                mount_use_sum[weapon_id] += mount_use_step[weapon_id]
            r.append(step_reward)

        if self.dr:
            intercept_reward_list, intercept_sum, terminal, intercept_list, blue_missile = \
                self.compute_intercept_reward_list_2()
        else:
            intercept_reward_list, intercept_sum, terminal, intercept_list, blue_missile = \
                self.compute_intercept_reward_list_1()

        r = [[x + y + z] for x, y, z in zip(r, intercept_reward_list, self.health_loss)]
        episode_reward = sum(np.array(r).squeeze(1))

        average_impact_dist = self.all_impact_dist / 100000

        return r, episode_reward, intercept_sum, terminal, mount_use_sum,\
            intercept_list, average_impact_dist, blue_missile

    def mount_remain(self):
        step_remain_mount = [0, 0, 0, 0]
        for al_id in self.blue_ship:
            al_unit_health = self.get_health_agent(al_id)
            self.now_health[al_id] = al_unit_health
            if not self.death_tracker_ally[al_id]:
                if al_unit_health == 0:
                    self.death_tracker_ally[al_id] = 1
                else:
                    al_unit_mount = self.get_weapon_num(al_id)
                    step_remain_mount[0] += al_unit_mount[0]
                    step_remain_mount[1] += al_unit_mount[1]
                    step_remain_mount[2] += al_unit_mount[2]
                    step_remain_mount[3] += al_unit_mount[3]

        return step_remain_mount

    def time_to_step(self, time):
        hour = time[:2]
        if hour[0] == "0":
            hour = hour[1]
        hour = int(hour)
        minute = time[-2:]
        if minute[0] == "0":
            minute = minute[1]
        minute = int(minute)
        time_interval = 30
        if time_interval == 15:
            if 17 > hour >= 13:
                time_step = (hour - 13) * 4
                if minute < 15:
                    return time_step  # 0
                elif 15 <= minute < 30:
                    return time_step + 1
                elif 30 <= minute < 45:
                    return time_step + 2
                else:
                    return time_step + 3
            elif hour >= 17:
                return self.step_limit - 1
        if time_interval == 30:
            if hour >= 13:
                time_step = (hour - 13) * 2
                if minute < 30:
                    return time_step
                else:
                    return time_step + 1

    def compute_intercept_reward_list_1(self):
        path = "D:/Mozi/MoziServer/bin/Logs"
        lists = os.listdir(path)
        lists.sort(key=lambda fn: os.path.getmtime(path + "/" + fn))
        file_newest_path = os.path.join(path, lists[-1])
        wasted_file_path = os.path.join(path, lists[0])
        os.remove(wasted_file_path)

        f = open(file_newest_path, encoding='utf-8')

        intercept_reward_list = [0] * self.step_limit
        intercept_list = [0] * self.step_limit
        blue_missile = {}
        for key in range(self.step_limit):
            blue_missile[key] = []
        for line in f:
            if '蓝方: 发射单元' in line:
                time = line[13:18]
                step = self.time_to_step(time)
                if step >= self.step_limit:
                    step = self.step_limit - 1
                blue_missile_id_1 = line[-6:-3]
                for key, value in blue_missile.items():
                    if key == step:
                        value.append(blue_missile_id_1)

            if '蓝方: 武器' in line:
                if '爬升率' in line and '飞机' not in line:
                    time = line[13:18]
                    step = self.time_to_step(time)
                    if step >= self.step_limit:
                        step = self.step_limit - 1
                    intercept_reward_list[step] += 0.1
                    intercept_list[step] += 1
        intercept_reward_sum = sum(intercept_reward_list)
        intercept_sum = round(intercept_reward_sum * 10)
        terminal = False
        if intercept_sum == self.n_enemies:
            terminal = True

        f.close()
        os.remove(file_newest_path)
        return intercept_reward_list, intercept_sum, terminal, intercept_list, blue_missile

    def compute_intercept_reward_list_2(self):
        path = "D:/Mozi/MoziServer/bin/Logs"
        lists = os.listdir(path)
        lists.sort(key=lambda fn: os.path.getmtime(path + "/" + fn))
        file_newest_path = os.path.join(path, lists[-1])
        wasted_file_path = os.path.join(path, lists[0])
        os.remove(wasted_file_path)

        f = open(file_newest_path, encoding='utf-8')

        intercept_reward_list = [0] * self.step_limit
        intercept_list = [0] * self.step_limit
        blue_missile = {}

        for key in range(self.step_limit):
            blue_missile[key] = []

        intercept_sum = 0

        for line in f:
            if '蓝方: 发射单元' in line:
                time = line[13:18]
                step = self.time_to_step(time)
                if step >= self.step_limit:
                    step = self.step_limit - 1
                blue_missile_id_1 = line[-6:-3]
                idx = line.find(">")
                target = line[idx-3:idx]
                for key, value in blue_missile.items():
                    if key == step:
                        value.append([blue_missile_id_1, target])
                        break

            target_name = None
            t_end = 0
            if '蓝方: 武器' in line:
                if '爬升率' in line and '飞机' not in line:
                    time = line[13:18]
                    step = self.time_to_step(time)
                    if step >= self.step_limit:
                        step = self.step_limit - 1
                    idx = line.find('正')
                    blue_missile_id_2 = line[idx-3:idx]
                    end = False
                    for key, value in blue_missile.items():
                        for v in value:
                            if blue_missile_id_2 in v[0]:
                                target_name = v[1]
                                t_end = step
                                end = True
                                break
                        if end:
                            break

                    fire_ep = [0] * self.step_limit
                    fire_ep_sum = 0
                    for key in range(t_end + 1):
                        value = blue_missile[key]
                        for v in value:
                            if v[1] == target_name:
                                fire_ep[key] = 1
                                fire_ep_sum += 1

                    fire_ep_len = 0
                    fire_last_time = 0
                    for i, j in enumerate(fire_ep):
                        if j == 1:
                            fire_ep_len += 1
                            fire_last_time = i

                    single_intercept_reward = 0.1 * (0.9**fire_ep_len)
                    intercept_reward_list[fire_last_time] += single_intercept_reward
                    intercept_sum += 1
                    intercept_list[step] += 1

        terminal = False
        if intercept_sum == self.n_enemies:
            terminal = True

        f.close()
        os.remove(file_newest_path)
        return intercept_reward_list, intercept_sum, terminal, intercept_list, blue_missile
