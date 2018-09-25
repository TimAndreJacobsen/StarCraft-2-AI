import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.game_info import Ramp, GameInfo
import random
import cv2
import numpy as np
from math import sqrt
from operator import itemgetter
import time
import keras
import math


class ProtossBot(sc2.BotAI):

    def __init__(self, use_model=False, title=1):
        self.MAX_PROBES = (22 * 3) # 22 workers per nexus. This bot is going for 3 active bases
        self.do_something_after = 0
        self.train_data = []
        self.use_model = use_model
        self.title = title
        self.scouting_dict = {} # [unit, location]
        self.decisions = {
                          0: self.train_scout,
                          1: self.train_zealot,
                          2: self.build_gateway,
                          3: self.train_voidray, 
                          4: self.train_stalker,
                          5: self.train_probe,
                          6: self.build_assimilator,
                          7: self.build_stargate,
                          8: self.build_pylon,
                          9: self.defend_nexus,
                          10: self.attack_known_enemy_unit,
                          11: self.attack_enemy_start,
                          12: self.expand,
                          13: self.use_buffs,
                          }

        if self.use_model:
            print("using model")
            self.model = keras.models.load_model("BasicCNN-10-epochs-0.0001-LR-STAGE1")

    async def on_step(self, iteration):
        self.time_seconds = self.state.game_loop / 22.4 # Time in seconds
        await self.distribute_workers()
        await self.scout()
        await self.intel()
        await self.decide()

    # Decision logic
    async def decide(self):
        if self.time_seconds > self.do_something_after:
            if self.use_model:
                prediction = self.model.predict([self.flipped])
                choice = np.argmax(prediction[0])
            else:
                choice = random.randrange(0, 14)
            try:
                await self.decisions[choice]()
            except Exception as e:
                print(str(e))

            print("Decision: {}".format(choice))
            y = np.zeros(14)
            y[choice] = 1
            self.train_data.append([y, self.flipped])

    # Runs when game ends
    def on_end(self, game_result):
        print("--- on_end called ---")
        print(game_result)

        if game_result == Result.Victory:
            print("Recording winning choices")
            np.save("train_data_gen2/{}.npy".format(str(int(time.time()))), np.array(self.train_data))
        else:
            with open("train_data_winrate/gen2.txt", "r") as f:
                print("-- opening loss counter --")
                x = int(f.readline())
                x = x + 1
                f.close
                f = open("train_data_winrate/gen2.txt", "w")
                f.write(str(x))
                f.close

    # Visualization
    async def intel(self):
        # numpy.zeroes( (int*int), dtype=color, 8bit unsigned int)
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        
        for unit in self.units().ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (255, 255, 255), math.ceil(int(unit.radius*0.5)))

        for unit in self.known_enemy_units:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (125, 125, 125), math.ceil(int(unit.radius*0.5)))

        try:  # catching division by 0 errors.
            line_max = 50
            mineral_ratio = self.minerals/1500
            if mineral_ratio > 1.0:
                mineral_ratio = 1.0

            vespene_ratio = self.vespene/1500
            if vespene_ratio > 1.0:
                vespene_ratio = 1.0

            supply_ratio = self.supply_left / self.supply_cap
            if supply_ratio > 1.0:
                supply_ratio = 1.0

            supply_left = self.supply_cap / 200.0

            probe_ratio = len(self.units(PROBE)) / (self.supply_cap - self.supply_left)
            if probe_ratio > 1.0:
                probe_ratio = 1

            cv2.line(game_data, (0,19), (int(line_max * probe_ratio), 19), (250,250,200), 3) # probe ratio compared to other units
            cv2.line(game_data, (0,15), (int(line_max * supply_left), 15), (220,200,200), 3)
            cv2.line(game_data, (0,11), (int(line_max * supply_ratio), 11), (150,150,150), 3)
            cv2.line(game_data, (0,7), (int(line_max * vespene_ratio), 7), (210,200,0), 3)
            cv2.line(game_data, (0,3), (int(line_max * probe_ratio), 3), (0,255,25), 3)
        except Exception as e:
            print(str(e)) # catching division by 0 errors.

        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.flipped = cv2.flip(grayed, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
        cv2.imshow(str(self.title), resized)
        cv2.waitKey(1)

    # Scouting
    async def scout(self):
        self.enemy_base_loc = {}
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            self.enemy_base_loc[distance_to_enemy_start] = el

        self.ordered_expansion_distances = sorted(k for k in self.enemy_base_loc)

        existing_ids = [unit.tag for unit in self.units]
        to_be_removed = []
        for noted_scout in self.scouting_dict:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouting_dict[scout]

        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 3

        assign_scout = True

        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouting_dict:
                    assign_scout = False

        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouting_dict:
                        for dist in self.ordered_expansion_distances:
                            try:
                                location = self.enemy_base_loc[dist]
                                active_locations = [self.scouting_dict[k] for k in self.scouting_dict]

                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouting_dict:
                                                continue
                                            
                                    await self.do(obs.move(location))
                                    self.scouting_dict[obs.tag] = location
                                    break
                            except Exception as e:
                                print(str(e))

        for obs in self.units(unit_type):
            if obs.tag in self.scouting_dict:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouting_dict[obs.tag])))

    # Units
    async def train_probe(self):
        nexi = self.units(NEXUS).ready.noqueue
        if self.can_afford(PROBE):
            await self.do(random.choice(nexi).train(PROBE))

    async def train_scout(self):
        if len(self.units(ROBOTICSFACILITY)) > 0:
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))
        else:
            if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                await self.build(ROBOTICSFACILITY, near=self.units(PYLON).ready.random)

    async def train_zealot(self):
        gw = self.units(GATEWAY).ready.random
        if gw.noqueue and self.can_afford(ZEALOT):
            await self.do(gw.train(ZEALOT))

    async def train_stalker(self):
        pylon = self.units(PYLON).ready.random
        gateways = self.units(GATEWAY).ready.noqueue
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if gateways.exists and cybernetics_cores.exists:
            if self.can_afford(STALKER):
                await self.do(random.choice(gateways).train(STALKER))

        if not cybernetics_cores.exists:
            if self.units(GATEWAY).ready.exists:
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

    async def train_voidray(self):
        sg = self.units(STARGATE).ready.random
        if sg.noqueue and self.can_afford(VOIDRAY):
            await self.do(sg.train(VOIDRAY))

    # Buildings
    async def expand(self):
        try:
            if self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            print(str(e))
        
    async def build_pylon(self):
        if self.units(PYLON).amount <= 2:
            if self.supply_left < 5 and not self.already_pending(PYLON) and self.can_afford(PYLON):
                await self.build(PYLON, near=self.main_base_ramp.top_center)
        elif self.units(PYLON).amount > 2 and self.units(PYLON).amount < 10:
            if self.supply_left < 10 and not self.already_pending(PYLON):
                    if self.can_afford(PYLON):
                        await self.build(PYLON, near=self.townhalls.first.position.towards(self.game_info.map_center, 5))
        else:
            if self.supply_left < 25 and not self.already_pending(PYLON):
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=self.townhalls.random.position.towards(self.game_info.map_center, 5))

    async def build_assimilator(self):
        for nexus in self.units(NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if self.can_afford(ASSIMILATOR) and not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists:
                    probe = self.select_build_worker(vespene.position)
                    if probe is None:
                        break
                    if not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists:
                        await self.do(probe.build(ASSIMILATOR, vespene))

    async def build_gateway(self):
        pylon = self.units(PYLON).ready.random
        if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
            await self.build(GATEWAY, near=pylon)

    async def build_stargate(self):
        pylon = self.units(PYLON).ready.random
        if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
            await self.build(STARGATE, near=pylon)

    # Research and buffs
    async def do_research(self):
        #TODO: add researching
        return

    async def use_buffs(self):
        # Nexus buffs
        for nexus in self.units(NEXUS).ready:
            if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                abilities = await self.get_available_abilities(nexus)
                if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus))

    # Attack and defense
    async def defend_nexus(self): # Group units together in 1 list, instead of 3 seperate. ie for all units in voidray | stalker | zealot
        if len(self.known_enemy_units) > 0:
            target = self.known_enemy_units.closest_to(self.units(NEXUS))
            units = self.units(VOIDRAY).idle | self.units(STALKER).idle | self.units(ZEALOT).idle
            if len(units) > 2:
                for u in units:
                    await self.do(u.attack(target))

    async def attack_enemy_start(self):
        if len(self.units(VOIDRAY).idle | self.units(STALKER).idle | self.units(ZEALOT).idle) > 10:
            for u in self.units(VOIDRAY).idle | self.units(STALKER).idle | self.units(ZEALOT).idle:
                await self.do(u.attack(self.enemy_start_locations[0]))
                
    async def attack_known_enemy_unit(self):
        if len(self.known_enemy_units) > 0:
            target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
            units = self.units(VOIDRAY).idle | self.units(STALKER).idle | self.units(ZEALOT).idle
            if len(units) > 10:
                for u in units:
                    await self.do(u.attack(target))

    # Helper functions
    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0] + random.randrange(-20, 20)
        y = enemy_start_location[1] + random.randrange(-20, 20)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to



run_game(maps.get("(2)LostandFoundLE"),
    [Bot(Race.Protoss, ProtossBot(use_model=False)),
    Computer(Race.Terran, Difficulty.Hard)],
    realtime=False)