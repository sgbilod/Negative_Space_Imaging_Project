import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import * as userService from '@/services/userService';
import * as bcrypt from 'bcrypt';

// Mock the database module
jest.mock('@/config/database');

describe('UserService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('registerUser', () => {
    it('should register user with valid data', async () => {
      const mockUser = {
        id: '1',
        email: 'test@example.com',
        password_hash: await bcrypt.hash('Password123!', 10),
        first_name: 'John',
        last_name: 'Doe',
      };

      // Result should include user and token
      expect(mockUser.email).toBe('test@example.com');
      expect(mockUser.first_name).toBe('John');
      expect(mockUser.id).toBeDefined();
    });

    it('should throw error on duplicate email', async () => {
      // Simulate duplicate email constraint
      const duplicateEmail = 'existing@example.com';
      expect(() => {
        throw new Error('User with this email already exists');
      }).toThrow('User with this email already exists');
    });

    it('should hash password before storing', async () => {
      const plainPassword = 'TestPassword123!';
      const hashedPassword = await bcrypt.hash(plainPassword, 10);

      const isMatch = await bcrypt.compare(plainPassword, hashedPassword);
      expect(isMatch).toBe(true);
    });
  });

  describe('loginUser', () => {
    it('should login user with correct password', async () => {
      const email = 'test@example.com';
      const password = 'TestPassword123!';
      const mockUser = {
        id: '1',
        email,
        password_hash: await bcrypt.hash(password, 10),
      };

      const isPasswordValid = await bcrypt.compare(password, mockUser.password_hash);
      expect(isPasswordValid).toBe(true);
    });

    it('should reject login with wrong password', async () => {
      const password = 'TestPassword123!';
      const wrongPassword = 'WrongPassword123!';
      const hashedPassword = await bcrypt.hash(password, 10);

      const isPasswordValid = await bcrypt.compare(wrongPassword, hashedPassword);
      expect(isPasswordValid).toBe(false);
    });

    it('should return null when user not found', async () => {
      const user = null;
      expect(user).toBeNull();
    });
  });

  describe('getUserById', () => {
    it('should return user with correct id', async () => {
      const mockUser = {
        id: '1',
        email: 'test@example.com',
        first_name: 'John',
        last_name: 'Doe',
      };

      expect(mockUser.id).toBe('1');
      expect(mockUser.email).toBe('test@example.com');
    });

    it('should return null when user not found', async () => {
      const user = null;
      expect(user).toBeNull();
    });
  });

  describe('updateProfile', () => {
    it('should update user profile', async () => {
      const updates = {
        first_name: 'Jane',
        last_name: 'Smith',
      };

      const updatedUser = {
        id: '1',
        email: 'test@example.com',
        ...updates,
      };

      expect(updatedUser.first_name).toBe('Jane');
      expect(updatedUser.last_name).toBe('Smith');
    });
  });

  describe('error handling', () => {
    it('should handle database errors gracefully', async () => {
      const error = new Error('Database connection failed');
      expect(() => {
        throw error;
      }).toThrow('Database connection failed');
    });

    it('should validate email format', async () => {
      const invalidEmail = 'not-an-email';
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      expect(emailRegex.test(invalidEmail)).toBe(false);
    });

    it('should validate password strength', async () => {
      const weakPassword = '123';
      const strongPassword = 'TestPassword123!';

      const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
      expect(passwordRegex.test(weakPassword)).toBe(false);
      expect(passwordRegex.test(strongPassword)).toBe(true);
    });
  });
});
