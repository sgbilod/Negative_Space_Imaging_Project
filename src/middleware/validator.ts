// Stub for request validator middleware
import { Request, Response, NextFunction } from 'express';
export function validateRequest(schema: any) {
  return (req: Request, res: Response, next: NextFunction) => {
    // No-op validation
    next();
  };
}
