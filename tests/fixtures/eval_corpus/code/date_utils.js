const formatterCache = new Map();

export function toZone(isoString, timeZone) {
  const instant = new Date(isoString);
  if (Number.isNaN(instant.getTime())) {
    throw new RangeError(`not an ISO-8601 date: ${isoString}`);
  }
  if (!formatterCache.has(timeZone)) {
    formatterCache.set(timeZone, new Intl.DateTimeFormat("en-CA", {
      timeZone, year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", hour12: false,
    }));
  }
  return formatterCache.get(timeZone).format(instant);
}
