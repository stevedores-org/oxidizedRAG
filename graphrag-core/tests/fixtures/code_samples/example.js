export function createCounter(start = 0) {
  let value = start;

  return {
    increment() {
      value += 1;
      return value;
    },
    current() {
      return value;
    }
  };
}

export async function fetchUserProfile(client, userId) {
  const response = await client.get(`/users/${userId}`);
  return response.data;
}

export class ReportService {
  constructor(client) {
    this.client = client;
  }

  async generate(userId) {
    const profile = await fetchUserProfile(this.client, userId);
    return `${profile.name}:${profile.id}`;
  }
}
