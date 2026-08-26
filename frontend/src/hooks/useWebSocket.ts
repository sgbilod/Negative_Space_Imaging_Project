/**
 * WebSocket Hook
 * Manages WebSocket connections
 */

import { useEffect, useState } from 'react';

export const useWebSocket = (url: string) => {
  const [data, setData] = useState<unknown>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    try {
      const ws = new WebSocket(url);

      ws.onopen = () => setIsConnected(true);
      ws.onmessage = (event) => setData(event.data);
      ws.onerror = () => setError(new Error('WebSocket error'));
      ws.onclose = () => setIsConnected(false);

      return () => ws.close();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('WebSocket connection failed'));
    }
  }, [url]);

  return { data, isConnected, error };
};
