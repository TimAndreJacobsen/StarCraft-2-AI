import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.game_info import Ramp, GameInfo
import random
import cv2
import numpy as np


class ProtossBot(sc2.BotAI):

    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 168 # From own testing in-game
        self.MAX_PROBES = (22 * 3) # 22 workers per nexus. This bot is going for 3 bases
        self.GAME_TIME = 0 # In minutes

    async def on_step(self, iteration):
        self.iteration = iteration

        if (self.iteration % self.ITERATIONS_PER_MINUTE) == 0:
            await self.chat_send("elapsed time: {}min".format(int(self.iteration / self.ITERATIONS_PER_MINUTE)))
            self.GAME_TIME = self.GAME_TIME + 1
        
        await self.distribute_workers()
        await self.use_buffs()
        await self.train_probe()
        await self.use_buffs()
        await self.build_pylon()
        await self.build_assimilator()
        await self.expand(iteration)
        await self.cybernetics_core()
        await self.build_gateway()
        await self.train_army()
        await self.attack(iteration)
        await self.defend()

        await self.intel()

    async def intel(self):
        # Map x,y coords reversed and stored as a touple in numpy(zeros)
        # numpy.zeroes( (int* int), dtype=color, 8bit unsigned int)
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        for nexus in self.units(NEXUS):
            nex_pos = nexus.position
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
            cv2.circle(game_data, (int(nex_pos[0]), int(nex_pos[1])), 10, (0, 255, 0), -1)

        flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(flipped, dsize=None, fx=2, fy=2)
        cv2.imshow('Intel', resized)
        cv2.waitKey(1)

    async def train_probe(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.units(PROBE).amount < (self.units(NEXUS).amount * 22) and self.units(PROBE).amount < self.MAX_PROBES:
                if self.can_afford(PROBE) and not self.already_pending(PROBE):
                    await self.do(nexus.train(PROBE))
        
            if self.GAME_TIME > 8:
                if self.units(PROBE).amount < (self.units(NEXUS).amount * 22) and self.units(PROBE).amount < self.MAX_PROBES + 30:
                    if self.can_afford(PROBE) and not self.already_pending(PROBE):
                        await self.do(nexus.train(PROBE))

    async def use_buffs(self):
        for nexus in self.units(NEXUS).ready:
            if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                abilities = await self.get_available_abilities(nexus)
                if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus))

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
                    await self.do(probe.build(ASSIMILATOR, vespene))

    async def expand(self, iteration):
        if self.units(NEXUS).amount == 1:
            if self.can_afford(NEXUS):
                await self.expand_now()
        elif self.units(NEXUS).amount == 2 and self.units(PROBE).amount > 30:
            if self.can_afford(NEXUS):
                await self.expand_now()
        elif len(self.units(NEXUS)) < ((self.iteration / self.ITERATIONS_PER_MINUTE) / 2):
            if self.can_afford(NEXUS) and not self.already_pending(NEXUS):
                await self.expand_now()
        
    async def cybernetics_core(self):
        if self.units(PYLON).ready.exists and self.units(CYBERNETICSCORE).amount < 1:
            pylon = self.units(PYLON).ready.random

            if not self.units(GATEWAY).exists or not self.already_pending(GATEWAY):
                if self.can_afford(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(GATEWAY).ready.exists and not self.already_pending(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)
        #TODO: add researching

    # If you have a pylon and expansion(state?)
    async def build_gateway(self):
        if self.units(PYLON).exists:
            pylon = self.units(PYLON).random

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

            elif self.units(GATEWAY).amount < 1:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < (self.iteration / self.ITERATIONS_PER_MINUTE):
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    async def train_army(self):
        for stargate in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 2:
                await self.do(stargate.train(VOIDRAY))

    async def defend(self):
        # Stalker
        if self.units(STALKER).amount > 0:
            for unit in self.units(STALKER).idle:
                if len(self.known_enemy_units) > 1:
                    await self.do(unit.attack(random.choice(self.known_enemy_units)))
        # VoidRay
        if self.units(VOIDRAY).amount > 0:
            for unit in self.units(VOIDRAY).idle:
                if len(self.known_enemy_units) > 1:
                    await self.do(unit.attack(random.choice(self.known_enemy_units)))
        # Zealot
        if self.units(ZEALOT).amount > 0:
            for unit in self.units(ZEALOT).idle:
                if len(self.known_enemy_units) > 1:
                    await self.do(unit.attack(random.choice(self.known_enemy_units)))

    def find_target(self, state):
        return self.enemy_start_locations[0]

    async def attack(self, iteration):
        if self.GAME_TIME <= 5:

            aggressive_units = {STALKER: [0, 0],
                                VOIDRAY: [5, 1]}

            for UNIT in aggressive_units: # Attack the enemy
                if self.units(UNIT).amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
                    for u in self.units(UNIT).idle:
                        await self.do(u.attack(self.find_target(self.state)))
                elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
                    if len(self.known_enemy_units) > 0:
                        for u in self.units(UNIT).idle:
                            await self.do(u.attack(random.choice(self.known_enemy_units)))
                            
        else:
            if iteration % self.ITERATIONS_PER_MINUTE == 0:
                if self.units(VOIDRAY).amount > 5:
                    for unit in self.units(STALKER).idle | self.units(VOIDRAY).idle:
                        await self.do(unit.attack(self.find_target(self.state)))

run_game(maps.get("(2)LostandFoundLE"),
    [Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Terran, Difficulty.Hard)],
    realtime=False)