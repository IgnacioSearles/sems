use std::io::{self, Write};

pub struct Row {
    pub name: String,
    pub email: String,
    pub balance_cents: i64,
}

/// Quotes a field when it contains the delimiter, quotes or line breaks (RFC 4180).
fn escape_field(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

pub fn export(rows: &[Row], mut out: impl Write) -> io::Result<()> {
    writeln!(out, "name,email,balance")?;
    for row in rows {
        let balance = format!("{}.{:02}", row.balance_cents / 100, row.balance_cents % 100);
        writeln!(out, "{},{},{}", escape_field(&row.name), escape_field(&row.email), balance)?;
    }
    Ok(())
}
