
import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				// CareerSync Premium colors
				navy: {
					50: '#e6f1fe',
					100: '#cce3fd',
					200: '#99c7fb',
					300: '#66abf9',
					400: '#338ff7',
					500: '#0466C8', // Electric blue
					600: '#0355a6',
					700: '#024483',
					800: '#023461',
					900: '#0A1128', // Deep navy
				},
				teal: {
					50: '#e6f9f9',
					100: '#ccf3f3',
					200: '#99e7e6',
					300: '#66dbda',
					400: '#33cfcd',
					500: '#33C1BF', // Soft teal
					600: '#2a9e9c',
					700: '#217c7a',
					800: '#195a59',
					900: '#123837',
				},
				silver: {
					50: '#fcfcfc',
					100: '#f9f9fa',
					200: '#f4f4f5',
					300: '#eeeff0',
					400: '#e9eaeb',
					500: '#E6E8E9', // Platinum silver
					600: '#bbbcbd',
					700: '#909193',
					800: '#666768',
					900: '#3c3c3d',
				},
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: { height: '0' },
					to: { height: 'var(--radix-accordion-content-height)' },
				},
				'accordion-up': {
					from: { height: 'var(--radix-accordion-content-height)' },
					to: { height: '0' },
				},
				'pulse-gentle': {
					'0%, 100%': { opacity: '1' },
					'50%': { opacity: '0.8' },
				},
				'float': {
					'0%, 100%': { transform: 'translateY(0)' },
					'50%': { transform: 'translateY(-10px)' },
				},
				'shimmer': {
					'0%': { backgroundPosition: '-500px 0' },
					'100%': { backgroundPosition: '500px 0' },
				},
				'fade-in': {
					'0%': { opacity: '0', transform: 'translateY(10px)' },
					'100%': { opacity: '1', transform: 'translateY(0)' },
				},
				'slide-up': {
					'0%': { transform: 'translateY(20px)', opacity: '0' },
					'100%': { transform: 'translateY(0)', opacity: '1' },
				},
				'slide-in-right': {
					'0%': { transform: 'translateX(20px)', opacity: '0' },
					'100%': { transform: 'translateX(0)', opacity: '1' },
				},
				'scale-in': {
					'0%': { transform: 'scale(0.95)', opacity: '0' },
					'100%': { transform: 'scale(1)', opacity: '1' },
				},
				'subtle-bounce': {
					'0%, 100%': { transform: 'translateY(0)' },
					'50%': { transform: 'translateY(-3px)' },
				},
				'progress-line': {
					'0%': { width: '0%' },
					'100%': { width: '100%' },
				},
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'pulse-gentle': 'pulse-gentle 3s ease-in-out infinite',
				'float': 'float 6s ease-in-out infinite',
				'shimmer': 'shimmer 2s infinite linear',
				'fade-in': 'fade-in 0.6s ease-out',
				'slide-up': 'slide-up 0.6s ease-out',
				'slide-in-right': 'slide-in-right 0.6s ease-out',
				'scale-in': 'scale-in 0.4s ease-out',
				'subtle-bounce': 'subtle-bounce 2s ease-in-out infinite',
				'progress-line': 'progress-line 2.5s ease-out',
			},
			fontFamily: {
				sans: ['Inter', 'SF Pro Display', 'system-ui', 'sans-serif'],
			},
			backgroundImage: {
				'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
				'glass-gradient': 'linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03))',
				'premium-gradient': 'linear-gradient(135deg, #0A1128, #0b1b47)',
				'card-gradient': 'linear-gradient(135deg, rgba(4, 102, 200, 0.08), rgba(51, 193, 191, 0.05))',
				'button-gradient': 'linear-gradient(to right, #0466C8, #0355a6)',
				'teal-gradient': 'linear-gradient(to right, #33C1BF, #2a9e9c)',
			},
			boxShadow: {
				'glass': '0 4px 30px rgba(0, 0, 0, 0.1)',
				'card': '0 10px 30px -5px rgba(10, 17, 40, 0.2)',
				'premium': '0 8px 30px rgba(4, 102, 200, 0.14)',
				'hover': '0 10px 40px -5px rgba(4, 102, 200, 0.3)',
				'button': '0 4px 10px rgba(4, 102, 200, 0.3)',
				'subtle': '0 2px 5px rgba(10, 17, 40, 0.08)',
			},
			spacing: {
				'18': '4.5rem',
				'22': '5.5rem',
			},
			letterSpacing: {
				'tight': '-0.01em',
				'tighter': '-0.02em',
				'super-tight': '-0.03em',
			},
			lineHeight: {
				'extra-tight': '1.1',
			},
			transitionDuration: {
				'400': '400ms',
			},
		}
	},
	plugins: [require("tailwindcss-animate")],
} satisfies Config;
