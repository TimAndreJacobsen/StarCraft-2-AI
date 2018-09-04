import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.game_info import Ramp, GameInfo
import random


class ProtossBot(sc2.BotAI):

    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_PROBES = (22 * 3) # 22 workers per nexus. This bot is going for 3 bases

    async def on_step(self, iteration):
        self.iteration = iteration
        
        await self.distribute_workers()
        await self.use_buffs()
        await self.train_probe()
        await self.use_buffs()
        await self.build_pylon()
        await self.build_assimilator()
        await self.expand()
        await self.cybernetics_core()
        await self.build_gateway()
        await self.train_army()
        await self.attack(iteration)
        await self.defend()

    async def train_probe(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.units(PROBE).amount < (self.units(NEXUS).amount * 22) and self.units(PROBE).amount < self.MAX_PROBES:
                if self.can_afford(PROBE) and not self.already_pending(PROBE):
                    await self.do(nexus.train(PROBE))

    async def use_buffs(self):
        for nexus in self.units(NEXUS).ready:
            if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                abilities = await self.get_available_abilities(nexus)
                if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus))

    async def build_pylon(self):
        if self.units(NEXUS).amount < 2:
            if self.supply_left < 5 and not self.already_pending(PYLON):
                nexi = self.units(NEXUS).ready
                if nexi.exists:
                    if self.can_afford(PYLON):
                        await self.build(PYLON, near=self.main_base_ramp.top_center)
        elif self.units(NEXUS).amount > 2:
            if self.supply_left < 15 and len(self.units(PYLON).not_ready) < 1:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=self.units(NEXUS).random)
        # if self.can_afford(PYLON) and not self.already_pending(PYLON):
        #     location = self.find_placement(PYLON, near=self.units(NEXUS).first)
        #     await self.build(PYLON, location)

    async def build_assimilator(self):
        if self.units(PYLON).amount <= self.units(ASSIMILATOR).amount:
            return
        for nexus in self.units(NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if self.can_afford(ASSIMILATOR) and not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists:
                    probe = self.select_build_worker(vespene.position)
                    if probe is None:
                        break
                    await self.do(probe.build(ASSIMILATOR, vespene))

    async def expand(self):
        if self.units(NEXUS).amount == 1 and self.units(PROBE).amount > 15:
            if self.can_afford(NEXUS):
                await self.expand_now()
        
        if self.units(NEXUS).amount == 2 and self.units(PROBE).amount > 30:
            if self.can_afford(NEXUS):
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

    async def build_gateway(self):
        if self.units(PYLON).ready.exists and self.units(NEXUS).amount > 1:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).amount < self.units(NEXUS).amount and (self.already_pending(CYBERNETICSCORE) or self.units(CYBERNETICSCORE).exists):
                if self.can_afford(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(STARGATE).amount < (self.units(NEXUS).amount):
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)

    async def train_army(self):
        for gateway in self.units(GATEWAY).ready.noqueue:
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STALKER) and self.supply_left > 2:
                    await self.do(gateway.train(STALKER))

        for stargate in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 2:
                await self.do(stargate.train(VOIDRAY))

    async def defend(self):
        #Stalkers
        if self.units(STALKER).amount > 0:
            for unit in self.units(STALKER).idle:
                if len(self.known_enemy_units) > 0:
                    await self.do(unit.attack(random.choice(self.known_enemy_units)))
        #Void Rays
        if self.units(VOIDRAY).amount > 0:
            for unit in self.units(VOIDRAY).idle:
                if len(self.known_enemy_units) > 0:
                    await self.do(unit.attack(random.choice(self.known_enemy_units)))

    def find_target(self, state):
        return self.enemy_start_locations[0]

    async def attack(self, iteration):
        if iteration % self.ITERATIONS_PER_MINUTE == 0:
            if self.units(STALKER).amount > 5 and self.units(VOIDRAY).amount > 1:
                if self.units(NEXUS).amount >= 2:
                    for unit in self.units(STALKER).idle | self.units(VOIDRAY).idle:
                        await self.do(unit.attack(self.find_target(self.state)))

run_game(maps.get("(2)LostandFoundLE"),
    [Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Terran, Difficulty.Hard)],
    realtime=False)