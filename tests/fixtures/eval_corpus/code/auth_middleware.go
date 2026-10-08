package middleware

import (
	"net/http"
	"strings"

	"github.com/golang-jwt/jwt/v5"
)

func RequireJWT(secret []byte, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header := r.Header.Get("Authorization")
		raw, found := strings.CutPrefix(header, "Bearer ")
		if !found {
			http.Error(w, "missing credentials", http.StatusUnauthorized)
			return
		}
		token, err := jwt.Parse(raw, func(t *jwt.Token) (any, error) { return secret, nil },
			jwt.WithValidMethods([]string{"HS256"}))
		if err != nil || !token.Valid {
			http.Error(w, "invalid credentials", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
