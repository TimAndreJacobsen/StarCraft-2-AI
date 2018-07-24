import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *
import random


class ProtossBot(sc2.BotAI):

    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.train_probe()
        await self.build_pylon()
        await self.build_assimilator()
        await self.expand()
        await self.build_gateway()
        await self.train_army()
        await self.attack()
        await self.defend()

    async def train_probe(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.units(PROBE).amount < (self.units(NEXUS).amount * 22):
                if self.can_afford(PROBE) and not self.already_pending(PROBE):
                    await self.do(nexus.train(PROBE))

    async def build_pylon(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexi = self.units(NEXUS).ready
            if nexi.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexi.first)
        
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
            if self.can_afford(NEXUS) and self.units(STALKER).amount > 20:
                await self.expand_now()
        
    async def build_gateway(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

            if self.units(NEXUS).amount > 1:
                if self.units(GATEWAY).amount < (self.units(NEXUS).amount * 3):
                    if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                        await self.build(GATEWAY, near=pylon)

    async def train_army(self):
        for gateway in self.units(GATEWAY).ready.noqueue:
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STALKER) and self.supply_left > 0:
                    await self.do(gateway.train(STALKER))

    async def defend(self):
        if self.units(STALKER).amount > 2:
            for s in self.units(STALKER).idle:
                if len(self.known_enemy_units) > 0:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        if self.units(STALKER).amount > 15:
            if self.units(NEXUS).amount >= 2:
                for s in self.units(STALKER).idle:
                    await self.do(s.attack(self.find_target(self.state)))

run_game(maps.get("(2)LostandFoundLE"),
    [Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Terran, Difficulty.Easy)],
    realtime=False)