import { sequelize } from '@/config/database';
import { User } from '@/models/User';
import { Image } from '@/models/Image';
import { AnalysisResult } from '@/models/AnalysisResult';
import bcrypt from 'bcrypt';

interface SeedOptions {
  usersCount?: number;
  imagesPerUser?: number;
  truncate?: boolean;
}

const userFactory = async (index: number, password: string = 'TestPassword123!') => {
  const hashedPassword = await bcrypt.hash(password, 10);
  return {
    email: `testuser${index}@example.com`,
    password_hash: hashedPassword,
    first_name: `Test${index}`,
    last_name: `User${index}`,
  };
};

const imageFactory = (userId: number, index: number) => {
  return {
    user_id: userId,
    filename: `test-image-${userId}-${index}.jpg`,
    original_filename: `test_image_${userId}_${index}.jpg`,
    file_size: 1024000 + Math.random() * 5000000, // 1MB to 6MB
    storage_path: `/uploads/user-${userId}/test-image-${index}.jpg`,
    mime_type: 'image/jpeg',
    width: 1920 + Math.random() * 2080,
    height: 1080 + Math.random() * 1440,
  };
};

const analysisFactory = (imageId: number) => {
  const negativeSpacePercentage = Math.random() * 100;
  return {
    image_id: imageId,
    negative_space_percentage: negativeSpacePercentage,
    positive_space_percentage: 100 - negativeSpacePercentage,
    regions_count: Math.floor(Math.random() * 50) + 5,
    processing_time_ms: Math.floor(Math.random() * 5000) + 500,
    confidence_score: Math.random() * 0.4 + 0.6, // 0.6 to 1.0
    status: 'completed',
    raw_data: {
      algorithm: 'canny_edge_detection',
      thresholds: { low: 100, high: 200 },
      contours_found: Math.floor(Math.random() * 100),
    },
    visualization_data: {
      heatmap: 'encoded_image_data',
      bounding_boxes: Array.from({ length: 5 }, (_, i) => ({
        id: i,
        x: Math.random() * 1920,
        y: Math.random() * 1080,
        width: Math.random() * 500,
        height: Math.random() * 500,
        confidence: Math.random(),
      })),
    },
  };
};

export const seed = async (options: SeedOptions = {}) => {
  const {
    usersCount = 3,
    imagesPerUser = 2,
    truncate = true,
  } = options;

  try {
    // Truncate existing data if requested
    if (truncate) {
      console.log('Truncating existing data...');
      await sequelize.query('TRUNCATE TABLE analysis_results RESTART IDENTITY CASCADE');
      await sequelize.query('TRUNCATE TABLE images RESTART IDENTITY CASCADE');
      await sequelize.query('TRUNCATE TABLE users RESTART IDENTITY CASCADE');
    }

    console.log(`Seeding ${usersCount} users...`);
    const users = [];
    for (let i = 1; i <= usersCount; i++) {
      const userData = await userFactory(i);
      const user = await User.create(userData);
      users.push(user);
      console.log(`✓ Created user: ${user.email}`);
    }

    console.log(`Seeding ${imagesPerUser} images per user...`);
    for (const user of users) {
      for (let i = 1; i <= imagesPerUser; i++) {
        const imageData = imageFactory(user.id, i);
        const image = await Image.create(imageData);

        // Create analysis result for the image
        const analysisData = analysisFactory(image.id);
        await AnalysisResult.create(analysisData);
        console.log(`✓ Created image and analysis: ${image.original_filename}`);
      }
    }

    console.log('✓ Seed completed successfully!');
    console.log(`Total users: ${usersCount}`);
    console.log(`Total images: ${usersCount * imagesPerUser}`);
    console.log(`Total analysis results: ${usersCount * imagesPerUser}`);
  } catch (error) {
    console.error('Error seeding database:', error);
    throw error;
  }
};

// Run seed if executed directly
if (require.main === module) {
  seed()
    .then(() => {
      console.log('Seed completed!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Seed failed:', error);
      process.exit(1);
    });
}
