# Stage 0: Build and compile the frontend, based on Node.js.
FROM node:18-slim as build-stage

WORKDIR /app

# Install required build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Add custom .npmrc for faster installs
RUN echo "network-timeout=100000\nprefer-offline=true\nprogress=false" > .npmrc

COPY package.json package-lock.json /app/

# Supposedly more reliable than npm install
RUN npm ci --quiet

COPY ./ /app/

ARG BACKEND_URL
ARG INSTITUTION

# Make sure these two required build args are provided.
RUN test -n "$BACKEND_URL"
RUN test -n "$INSTITUTION"

RUN VITE_INSTITUTION=$INSTITUTION \
    VITE_BACKEND_URL=$BACKEND_URL \
    npm run build

ARG HOST

CMD ["npm", "run", "dev", "--", "--port", "3000", "--host", "0.0.0.0"]

# Stage 1: Serve the built files in production, based on nginx.
FROM nginx:1.19.1 as prod-stage

COPY --from=build-stage /app/dist/ /var/www

COPY ./nginx.conf /etc/nginx/conf.d/default.conf

# CMD ["nginx -g 'daemon off;'"]
