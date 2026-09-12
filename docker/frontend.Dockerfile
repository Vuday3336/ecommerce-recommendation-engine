# syntax=docker/dockerfile:1.7
#
# Frontend image: build with Node, serve the static output with nginx.
#
# The runtime image contains no Node and no node_modules - just nginx and a few
# hundred kilobytes of built assets.

FROM node:20-alpine AS builder

WORKDIR /app
# Dependencies are copied and installed before the source, so a source-only
# change reuses the cached install layer.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY frontend/ ./
ARG VITE_API_BASE_URL=/api/v1
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build


FROM nginx:1.27-alpine AS runtime

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=20s --timeout=3s --retries=3 \
    CMD wget -q --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
