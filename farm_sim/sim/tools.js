'use strict';

/**
 * Machine-readable action catalogue.
 * Serves `/api/v1/schema` and can be fed straight into an LLM tool-calling
 * request (name / description / parameters).
 */

const TILE_ARG = {
  type: 'array',
  items: { type: 'integer' },
  minItems: 2,
  maxItems: 2,
  description: '[x, y] tile coordinates.',
};

const TOOLS = [
  {
    name: 'move',
    description: 'Move the farmer exactly one tile north/south/east/west.',
    parameters: {
      dir: { type: 'string', enum: ['up', 'down', 'left', 'right'], description: 'Direction to step.' },
    },
    required: [],
  },
  {
    name: 'move_to',
    description: 'Walk to a tile with A* pathfinding. Consumes 1 energy per tile.',
    parameters: {
      tile: TILE_ARG,
      steps: { type: 'integer', description: 'Optional maximum number of tiles to walk.' },
    },
    required: ['tile'],
  },
  {
    name: 'till',
    description: 'Till an empty plot so it can be planted. Costs 4 energy.',
    parameters: { tile: TILE_ARG },
    required: ['tile'],
  },
  {
    name: 'plant_crop',
    description: 'Plant a seed on a tilled, empty plot within reach.',
    parameters: {
      crop: { type: 'string', enum: ['wheat', 'rice', 'corn', 'tomato'] },
      tile: TILE_ARG,
    },
    required: ['crop'],
  },
  {
    name: 'water',
    // Preset-dependent: `applyAction` bills `presetConfig.water_per_action`
    // (4 L on every shipped preset). This text used to say 2 L, which the agent
    // then budgeted against, so it is now stated as the preset's constant.
    description: 'Water a plot from the watering can. Costs 4 L (the preset\'s water_per_action) and 2 energy.',
    parameters: { tile: TILE_ARG },
    required: [],
  },
  {
    name: 'collect_water',
    description: 'Fill the watering can from the tank. Stand next to the tank (3,10).',
    parameters: {
      liters: { type: 'number', description: 'Litres to take (default: fill the can).' },
    },
    required: [],
  },
  {
    name: 'harvest',
    description: 'Harvest a ripe crop into the silo.',
    parameters: { tile: TILE_ARG },
    required: [],
  },
  {
    name: 'clear',
    description: 'Remove a dead or unwanted crop so the plot can be reused.',
    parameters: { tile: TILE_ARG },
    required: [],
  },
  {
    name: 'weed',
    description: 'Remove weeds from a planted plot. Costs 2 energy.',
    parameters: { tile: TILE_ARG },
    required: [],
  },
  {
    name: 'treat',
    description: 'Treat a diseased plot for $4 and 2 energy.',
    parameters: { tile: TILE_ARG },
    required: [],
  },
  {
    name: 'fertilize',
    description: 'Raise soil fertility for $6 and 3 energy.',
    parameters: { tile: TILE_ARG },
    required: [],
  },
  {
    name: 'scan',
    description: 'Read soil moisture, crop stage and water needs of nearby plots.',
    parameters: {
      radius: { type: 'integer', description: 'Search radius in tiles (0-8, default 2).' },
    },
    required: [],
  },
  {
    name: 'inspect',
    description: 'Detailed status of one tile or plot.',
    parameters: { tile: TILE_ARG },
    required: [],
  },
  {
    name: 'buy',
    description: 'Buy seeds at the silo (5,4).',
    parameters: {
      crop: { type: 'string', enum: ['wheat', 'rice', 'corn', 'tomato'] },
      qty: { type: 'integer', description: 'Number of seeds.' },
    },
    required: ['crop'],
  },
  {
    name: 'sell',
    description: 'Sell stored crops at the silo (5,4).',
    parameters: {
      crop: { type: 'string', enum: ['wheat', 'rice', 'corn', 'tomato'] },
      kg: { type: 'number', description: 'Kilograms to sell (default: all).' },
      contract: { type: 'string', description: 'Optional contract id for a better price.' },
    },
    required: ['crop'],
  },
  {
    name: 'dispatch',
    description: 'Send a rover to water dry plots, charge, or recall it.',
    parameters: {
      robot: { type: 'string', description: 'Rover id (default: first rover).' },
      task: { type: 'string', enum: ['water', 'charge', 'recall'] },
    },
    required: ['task'],
  },
  {
    name: 'rest',
    description: 'Rest at the farmhouse (2,6) to restore 35 energy.',
    parameters: {},
    required: [],
  },
  {
    name: 'advance',
    description: 'Fast-forward the simulation (useful for headless runs).',
    parameters: {
      minutes: { type: 'integer', description: 'Simulated minutes to skip.' },
      ticks: { type: 'integer', description: 'Ticks to skip (1 tick = 12 min).' },
    },
    required: [],
  },
];

const BY_NAME = Object.fromEntries(TOOLS.map((tool) => [tool.name, tool]));

module.exports = { TOOLS, BY_NAME };
