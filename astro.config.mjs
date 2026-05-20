import { defineConfig } from 'astro/config';

export default defineConfig({
  site: process.env.SITE ?? 'https://songy1nan.github.io',
  base: process.env.BASE_PATH ?? '/blog/',
});
