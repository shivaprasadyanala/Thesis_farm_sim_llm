'use strict';

const MAX_EVENTS = 250;

/**
 * Append an event to the world log.
 * The log is what the HUD ticker and the agent's `events` field read from,
 * so every state change that matters to a player must emit one.
 */
function pushEvent(world, type, message, data) {
  const evt = {
    tick: world.tick,
    day: world.time.day,
    minute: world.time.minute,
    type,
    message,
  };
  if (data !== undefined) evt.data = data;
  world.events.push(evt);
  if (world.events.length > MAX_EVENTS) {
    world.events.splice(0, world.events.length - MAX_EVENTS);
  }
  return evt;
}

function recentEvents(world, count = 12) {
  return world.events.slice(Math.max(0, world.events.length - count));
}

module.exports = { pushEvent, recentEvents, MAX_EVENTS };
