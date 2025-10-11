import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [
    react(),
    mode === 'development',
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      // Completely replace Supabase with our mock
      "@supabase/supabase-js": path.resolve(__dirname, "./src/integrations/supabase/mock.ts"),
    },
  },
  define: {
    // Disable Supabase in build
    'process.env.SUPABASE_DISABLED': '"true"',
  },
  optimizeDeps: {
    // Exclude Supabase from dependency optimization
    exclude: ['@supabase/supabase-js']
  },
}));
