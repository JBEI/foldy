# Stage 0, "build-stage", based on Node.js, to build and compile the frontend
FROM node:16 as build-stage

WORKDIR /app

COPY package.json package-lock.json /app/

RUN npm install

COPY ./ /app/

RUN npm run-script build

FROM nginx:1.19.1

COPY --from=build-stage /app/build/ /var/www

COPY ./nginx.conf /etc/nginx/conf.d/default.conf

# CMD ["nginx -g 'daemon off;'"]
