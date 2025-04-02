// Repository: millbj92/synctl
// File: internal/database/open_db_connection.go

package database

import (
	"os"

	"github.com/jmoiron/sqlx"
	"github.com/millbj92/synctl/internal/queries"
)

// Queries struct for collect all app queries.
type Queries struct {
	*queries.UserQueries // load queries from User model
	//*queries.BookQueries // load queries from Book model
}

// Connect func for opening database connection.
func Connect() (*Queries, error) {
	// Define Database connection variables.
	var (
		db  *sqlx.DB
		err error
	)

	// Get DB_TYPE value from .env file.
	dbType := os.Getenv("DB_TYPE")

	// Define a new Database connection with right DB type.
	switch dbType {
	case "pgx":
		db, err = PostgreSQLConnection()
	case "mysql":
		db, err = MysqlConnection()
	}

	if err != nil {
		return nil, err
	}

	return &Queries{
		// Set queries from models:
		UserQueries: &queries.UserQueries{DB: db}, // from User model
		//BookQueries: &queries.BookQueries{DB: db}, // from Book model
	}, nil
}
