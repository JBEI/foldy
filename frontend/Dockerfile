# Stage 0, "build-stage", based on Node.js, to build and compile the frontend
FROM node:16 as build-stage

WORKDIR /app

COPY package.json package-lock.json /app/

RUN npm install

COPY ./ /app/

ARG BACKEND_URL
ARG INSTITUTION

# Make sure these two required build args are provided.
RUN test -n "$BACKEND_URL"
RUN test -n "$INSTITUTION"

RUN REACT_APP_INSTITUTION=$INSTITUTION \
    REACT_APP_BACKEND_URL=$BACKEND_URL \
    npm run-script build

FROM nginx:1.19.1

COPY --from=build-stage /app/build/ /var/www

COPY ./nginx.conf /etc/nginx/conf.d/default.conf

# CMD ["nginx -g 'daemon off;'"]
