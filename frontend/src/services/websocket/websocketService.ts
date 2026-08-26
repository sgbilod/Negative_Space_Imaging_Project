/**
 * WebSocket Service
 * Manages WebSocket connections
 */

export const websocketService = {
  connect: (url: string): WebSocket => {
    return new WebSocket(url);
  },
};
