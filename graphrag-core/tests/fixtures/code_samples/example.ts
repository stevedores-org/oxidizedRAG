export interface Repository<T> {
  getById(id: string): Promise<T | null>;
  list(): Promise<T[]>;
}

export interface User {
  id: string;
  name: string;
}

export class InMemoryRepository<T extends { id: string }> implements Repository<T> {
  private values: Map<string, T> = new Map();

  async getById(id: string): Promise<T | null> {
    return this.values.get(id) ?? null;
  }

  async list(): Promise<T[]> {
    return Array.from(this.values.values());
  }

  save(item: T): void {
    this.values.set(item.id, item);
  }
}
