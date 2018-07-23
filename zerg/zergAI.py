import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.data import race_townhalls
import random

class DoubleBot(sc2.BotAI):

    async def on_step(self, iteration):

        forces = self.units(ZERGLING)
        larvae = self.units(LARVA)
        iter = iteration

        await self.distribute_workers()
        await self.train_drones(larvae)
        await self.train_overlords(larvae, iter)
        await self.train_queen()
        await self.build_extractor()
        await self.expand()
        await self.build_spawning_pool()
        await self.train_zerglings(larvae)
        await self.attack()

    async def train_drones(self, larvae):
        if self.units(DRONE).amount <= 20:
            if  self.supply_left > 1:
                if self.can_afford(DRONE) and larvae.exists:
                    await self.do(larvae.random.train(DRONE))
            else: # if we are close to supplycap, prioritize overlord
                if self.already_pending(OVERLORD):
                    if self.can_afford(DRONE) and larvae.exists and self.supply_left > 0:
                        await self.do(larvae.random.train(DRONE))
        else:
            if self.units(DRONE).amount < (self.units(HATCHERY).amount * 25):
                if self.already_pending(OVERLORD) and self.already_pending(ZERGLING):
                    if self.can_afford(DRONE) and larvae.exists and self.supply_left > 0:
                        await self.do(larvae.random.train(DRONE))

    async def train_overlords(self, larvae, iter):
        if iter < 5:    
            if self.supply_left < 3 and not self.already_pending(OVERLORD):
                if self.can_afford(OVERLORD) and larvae.exists:
                    await self.do(larvae.random.train(OVERLORD))
        else:
            if self.supply_left < 10 and not self.already_pending(OVERLORD):
                if self.can_afford(OVERLORD) and larvae.exists:
                    await self.do(larvae.random.train(OVERLORD))
                    return

    async def train_queen(self):
        if self.units(SPAWNINGPOOL).ready.exists:
            if not self.units(QUEEN).exists and self.townhalls.first.is_ready and self.townhalls.first.noqueue and not self.already_pending(QUEEN):
                if self.can_afford(QUEEN):
                    await self.do(self.townhalls.first.train(QUEEN))


    async def build_extractor(self):
        if self.units(EXTRACTOR).amount >= 0:
            if not self.units(SPAWNINGPOOL).exists or not self.already_pending(SPAWNINGPOOL):
                return
            for hatchery in self.units(HATCHERY).ready:
                vespenes = self.state.vespene_geyser.closer_than(15.0, hatchery)
                for vespene in vespenes:
                    if not self.can_afford(EXTRACTOR):
                        break
                    drone = self.select_build_worker(vespene.position)
                    if drone is None:
                        break
                    if not self.units(EXTRACTOR).closer_than(1.0, vespene).exists and not self.already_pending(EXTRACTOR):
                        await self.do(drone.build(EXTRACTOR, vespene))
        else:
            for hatchery in self.units(HATCHERY).ready:
                vespenes = self.state.vespene_geyser.closer_than(15.0, hatchery)
                for vespene in vespenes:
                    if not self.can_afford(EXTRACTOR):
                        break
                    drone = self.select_build_worker(vespene.position)
                    if drone is None:
                        break
                    if not self.units(EXTRACTOR).closer_than(1.0, vespene).exists and not self.already_pending(EXTRACTOR):
                        await self.do(drone.build(EXTRACTOR, vespene))

    async def expand(self):
        if self.units(HATCHERY).amount < 2 and self.can_afford(HATCHERY):
            await self.expand_now()

    async def build_spawning_pool(self):
        if not self.units(SPAWNINGPOOL).exists and not self.already_pending(SPAWNINGPOOL):
            if self.can_afford(SPAWNINGPOOL):
                await self.build(SPAWNINGPOOL, near=self.townhalls.first)

    async def train_zerglings(self, larvae):
        if self.units(SPAWNINGPOOL).ready.exists:
            if larvae.exists and self.can_afford(ZERGLING) and self.supply_left > 0 and not self.already_pending(QUEEN):
                await self.do(larvae.random.train(ZERGLING))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            self.enemy_start_locations[0]

    async def attack(self):
        if self.units(ZERGLING).amount > 20:
            for ling in self.units(ZERGLING).idle:
                await self.do(ling.attack(self.find_target(self.state)))

        elif self.units(ZERGLING).amount > 0:
            if len(self.known_enemy_units) > 0:
                for ling in self.units(ZERGLING):
                    await self.do(ling.attack(random.choice(self.known_enemy_units)))


run_game(maps.get("(2)CatalystLE"), 
    [Bot(Race.Zerg, DoubleBot()), 
    Computer(Race.Terran, Difficulty.Easy)],
    realtime=True)