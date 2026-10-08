export async function fetchWithBackoff(url: string, attempts = 5): Promise<Response> {
  let delayMs = 250;
  for (let attempt = 1; ; attempt++) {
    try {
      const response = await fetch(url);
      if (response.status < 500) return response;
      throw new Error(`server error ${response.status}`);
    } catch (error) {
      if (attempt >= attempts) throw error;
      const jitter = Math.random() * delayMs * 0.2;
      await new Promise((resolve) => setTimeout(resolve, delayMs + jitter));
      delayMs *= 2;
    }
  }
}
