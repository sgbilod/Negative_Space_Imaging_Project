import { Model, DataTypes, Optional } from 'sequelize';
import { sequelize } from '../database/connection';
import bcrypt from 'bcrypt';
import { config } from '../config';

/**
 * User attributes interface
 */
export interface UserAttributes {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  role: 'admin' | 'doctor' | 'technician' | 'researcher' | 'patient';
  isActive: boolean;
  lastLogin: Date | null;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * User creation attributes interface (optional fields during creation)
 */
export interface UserCreationAttributes extends Optional<UserAttributes, 'id' | 'isActive' | 'lastLogin' | 'createdAt' | 'updatedAt'> {}

/**
 * User model class
 */
export class User extends Model<UserAttributes, UserCreationAttributes> implements UserAttributes {
  public id!: string;
  public firstName!: string;
  public lastName!: string;
  public email!: string;
  public password!: string;
  public role!: 'admin' | 'doctor' | 'technician' | 'researcher' | 'patient';
  public isActive!: boolean;
  public lastLogin!: Date | null;
  public createdAt!: Date;
  public updatedAt!: Date;

  /**
   * Compare provided password with stored hashed password
   */
  public async comparePassword(password: string): Promise<boolean> {
    return bcrypt.compare(password, this.password);
  }

  /**
   * Returns user data without sensitive information
   */
  public toJSON(): Omit<UserAttributes, 'password'> {
    const values = { ...this.get() };
    delete values.password;
    return values as Omit<UserAttributes, 'password'>;
  }
}

/**
 * Initialize User model
 */
User.init(
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    firstName: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    lastName: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    email: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: true,
      validate: {
        isEmail: true,
      },
    },
    password: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    role: {
      type: DataTypes.ENUM('admin', 'doctor', 'technician', 'researcher', 'patient'),
      allowNull: false,
      defaultValue: 'researcher',
    },
    isActive: {
      type: DataTypes.BOOLEAN,
      allowNull: false,
      defaultValue: true,
    },
    lastLogin: {
      type: DataTypes.DATE,
      allowNull: true,
    },
    createdAt: {
      type: DataTypes.DATE,
      allowNull: false,
      defaultValue: DataTypes.NOW,
    },
    updatedAt: {
      type: DataTypes.DATE,
      allowNull: false,
      defaultValue: DataTypes.NOW,
    },
  },
  {
    sequelize,
    modelName: 'User',
    tableName: 'users',
    timestamps: true,
    hooks: {
      beforeCreate: async (user: User) => {
        // Hash password before creating user
        if (user.password) {
          user.password = await bcrypt.hash(
            user.password,
            config.security.bcryptSaltRounds
          );
        }
      },
      beforeUpdate: async (user: User) => {
        // Hash password if it's being updated
        if (user.changed('password')) {
          user.password = await bcrypt.hash(
            user.password,
            config.security.bcryptSaltRounds
          );
        }
      },
    },
  }
);

export default User;
