const env = {
  isDevelopment: process.env.NODE_ENV === 'development',
  isProduction: process.env.NODE_ENV === 'production',
};

export default env;

// Base path for code-server proxy support
export const BASE_PATH = '/absproxy/3000';

export type UserConfig = {
  isTelemetryEnabled: boolean;
  telemetryKey: string;
  telemetryHost: string;
  userUUID: string;
};

// Get the user configuration with basePath support
export const getUserConfig = async (): Promise<UserConfig> => {
  const config = await fetch('/absproxy/3000/api/config').then((res) =>
    res.json(),
  );
  const decodedTelemetryKey = Buffer.from(
    config.telemetryKey,
    'base64',
  ).toString();
  return { ...config, telemetryKey: decodedTelemetryKey };
};
