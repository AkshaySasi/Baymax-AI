/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        "baymax-blue": "#3B82F6",
        "baymax-gray": "#F3F4F6",
      },
    },
  },
  plugins: [],
};

