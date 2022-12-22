FROM node:16.17.1
WORKDIR /app
COPY package*.json .
# # -quiet
RUN npm install
COPY . .
CMD npm run start