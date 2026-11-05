# CreditLens frontend — Next.js 15 web demo on port 3003.
# NEXT_PUBLIC_API_URL is inlined at BUILD time, so pass it as a build arg; the
# variable has no effect on a running container. With no value the image is
# demo-only and calls no engine. If the configured backend is unreachable the UI
# falls back to the in-browser demo model.
FROM node:20-slim

WORKDIR /app

# Install deps first for layer caching.
COPY package.json package-lock.json ./
RUN npm ci

# App source.
COPY . .

# NEXT_PUBLIC_* must be present at build time to be inlined into the bundle.
ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_SITE_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_SITE_URL=$NEXT_PUBLIC_SITE_URL

RUN npm run build

EXPOSE 3003

CMD ["npm", "run", "start"]
