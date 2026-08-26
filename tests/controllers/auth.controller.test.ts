import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { Request, Response } from 'express';
import { AuthController } from '../../controllers/auth.controller';
import { User } from '../../models/User';
import { Security } from '../../models/Security';
import { config } from '../../config';
import { ApiError } from '../../middleware/errorHandler';

// Mock the models and other dependencies
jest.mock('../../models/User');
jest.mock('../../models/Security');
jest.mock('jsonwebtoken');
jest.mock('bcrypt');
jest.mock('../../config');

describe('AuthController', () => {
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;
  let mockNext: jest.Mock;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock request and response objects
    mockRequest = {
      body: {},
      ip: '127.0.0.1',
      headers: {
        'user-agent': 'test-agent',
      },
      user: undefined,
    };
    
    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn(),
    };
    
    mockNext = jest.fn();
    
    // Mock config
    (config.security as any) = {
      jwtSecret: 'test-secret',
      jwtExpiresIn: '1h',
      bcryptSaltRounds: 10,
    };
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  describe('register', () => {
    it('should create a new user and return 201 status', async () => {
      // Setup test data
      mockRequest.body = {
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@example.com',
        password: 'Password123!',
        role: 'researcher',
      };
      
      // Mock User.findOne to return null (user doesn't exist)
      (User.findOne as jest.Mock).mockResolvedValue(null);
      
      // Mock User.create to return a new user
      const mockUser = {
        id: '123',
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@example.com',
        role: 'researcher',
        toJSON: jest.fn().mockReturnValue({
          id: '123',
          firstName: 'John',
          lastName: 'Doe',
          email: 'john.doe@example.com',
          role: 'researcher',
        }),
      };
      (User.create as jest.Mock).mockResolvedValue(mockUser);
      
      // Mock Security.createAuditLog
      (Security.createAuditLog as jest.Mock).mockResolvedValue({});
      
      // Call the register method
      await AuthController.register(
        mockRequest as Request,
        mockResponse as Response,
        mockNext
      );
      
      // Assertions
      expect(User.findOne).toHaveBeenCalledWith({
        where: { email: 'john.doe@example.com' },
      });
      expect(User.create).toHaveBeenCalledWith({
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@example.com',
        password: 'Password123!',
        role: 'researcher',
      });
      expect(Security.createAuditLog).toHaveBeenCalled();
      expect(mockResponse.status).toHaveBeenCalledWith(201);
      expect(mockResponse.json).toHaveBeenCalledWith({
        status: 'success',
        data: {
          user: expect.objectContaining({
            id: '123',
            firstName: 'John',
            lastName: 'Doe',
            email: 'john.doe@example.com',
            role: 'researcher',
          }),
        },
      });
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should return 409 if user already exists', async () => {
      // Setup test data
      mockRequest.body = {
        firstName: 'John',
        lastName: 'Doe',
        email: 'existing@example.com',
        password: 'Password123!',
      };
      
      // Mock User.findOne to return an existing user
      (User.findOne as jest.Mock).mockResolvedValue({
        id: '123',
        email: 'existing@example.com',
      });
      
      // Call the register method
      await AuthController.register(
        mockRequest as Request,
        mockResponse as Response,
        mockNext
      );
      
      // Assertions
      expect(User.findOne).toHaveBeenCalledWith({
        where: { email: 'existing@example.com' },
      });
      expect(User.create).not.toHaveBeenCalled();
      expect(mockNext).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 409,
          message: 'User with this email already exists',
        })
      );
    });
  });

  describe('login', () => {
    it('should login user and return JWT token', async () => {
      // Setup test data
      mockRequest.body = {
        email: 'john.doe@example.com',
        password: 'Password123!',
      };
      
      // Mock User.findOne to return a user
      const mockUser = {
        id: '123',
        email: 'john.doe@example.com',
        role: 'researcher',
        isActive: true,
        comparePassword: jest.fn().mockResolvedValue(true),
        lastLogin: null,
        save: jest.fn().mockResolvedValue(true),
        toJSON: jest.fn().mockReturnValue({
          id: '123',
          email: 'john.doe@example.com',
          role: 'researcher',
        }),
      };
      (User.findOne as jest.Mock).mockResolvedValue(mockUser);
      
      // Mock jwt.sign
      (jwt.sign as jest.Mock).mockReturnValue('mock-token');
      
      // Mock Security.createAuditLog
      (Security.createAuditLog as jest.Mock).mockResolvedValue({});
      
      // Call the login method
      await AuthController.login(
        mockRequest as Request,
        mockResponse as Response,
        mockNext
      );
      
      // Assertions
      expect(User.findOne).toHaveBeenCalledWith({
        where: { email: 'john.doe@example.com' },
      });
      expect(mockUser.comparePassword).toHaveBeenCalledWith('Password123!');
      expect(mockUser.save).toHaveBeenCalled();
      expect(jwt.sign).toHaveBeenCalledWith(
        { id: '123', email: 'john.doe@example.com', role: 'researcher' },
        'test-secret',
        { expiresIn: '1h' }
      );
      expect(Security.createAuditLog).toHaveBeenCalled();
      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith({
        status: 'success',
        data: {
          user: expect.any(Object),
          token: 'mock-token',
          expiresIn: '1h',
        },
      });
    });

    it('should return 401 for invalid credentials', async () => {
      // Setup test data
      mockRequest.body = {
        email: 'john.doe@example.com',
        password: 'WrongPassword',
      };
      
      // Mock User.findOne to return a user
      const mockUser = {
        id: '123',
        email: 'john.doe@example.com',
        role: 'researcher',
        isActive: true,
        comparePassword: jest.fn().mockResolvedValue(false),
      };
      (User.findOne as jest.Mock).mockResolvedValue(mockUser);
      
      // Mock Security.createAuditLog
      (Security.createAuditLog as jest.Mock).mockResolvedValue({});
      
      // Call the login method
      await AuthController.login(
        mockRequest as Request,
        mockResponse as Response,
        mockNext
      );
      
      // Assertions
      expect(mockUser.comparePassword).toHaveBeenCalledWith('WrongPassword');
      expect(Security.createAuditLog).toHaveBeenCalledWith(
        expect.objectContaining({
          userId: '123',
          action: 'LOGIN',
          outcome: 'FAILURE',
        })
      );
      expect(mockNext).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 401,
          message: 'Invalid credentials',
        })
      );
    });

    it('should return 403 for inactive account', async () => {
      // Setup test data
      mockRequest.body = {
        email: 'inactive@example.com',
        password: 'Password123!',
      };
      
      // Mock User.findOne to return an inactive user
      const mockUser = {
        id: '123',
        email: 'inactive@example.com',
        role: 'researcher',
        isActive: false,
      };
      (User.findOne as jest.Mock).mockResolvedValue(mockUser);
      
      // Call the login method
      await AuthController.login(
        mockRequest as Request,
        mockResponse as Response,
        mockNext
      );
      
      // Assertions
      expect(mockNext).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 403,
          message: 'Account is disabled',
        })
      );
    });
  });

  // Additional tests for other methods would follow the same pattern
});
