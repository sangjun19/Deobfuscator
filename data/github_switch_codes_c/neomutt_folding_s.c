// Repository: neomutt/folding
// File: s.c

/**
 * cb_format_str - Create the string to show in the sidebar
 * @dest:        Buffer in which to save string
 * @destlen:     Buffer length
 * @flags:       Format flags, e.g. MUTT_FORMAT_OPTIONAL
 *
 * cb_format_str is a callback function for mutt_FormatString.  It understands
 * six operators. '%B' : Mailbox name, '%F' : Number of flagged messages,
 */
static const char *cb_format_str(char *dest, size_t destlen, size_t col, int cols,
                                 char op, const char *src, const char *prefix,
                                 const char *ifstring, const char *elsestring,
                                 unsigned long data, format_flag flags)
{
  SBENTRY *sbe = (SBENTRY *) data;
  unsigned int optional;
  char fmt[STRING];

  switch (op)
  {
    case 'B':
      /* This comment
       * deserves
       * more than
       * one line
       */
      mutt_format_s(dest, destlen, prefix, sbe->box);
      break;
  }

  return src;
}


/**
 * cb_format_str - Create the string to show in the sidebar
 */
static const char *cb_format_str(char *dest, size_t destlen, size_t col, int cols)
{
  return src;
}

/**
 * struct nm_hdrdata - NotMuch data attached to an email
 *
 * This stores all the NotMuch data associated with an email.
 *
 * @sa HEADER#data, MUTT_MBOX
 */
struct nm_hdrdata
{
  char *folder; /**< Location of the email */
  char *tags;
  char *tags_transformed;
  struct nm_hdrtag *tag_list;
  char *oldpath;
  char *virtual_id; /**< Unique NotMuch Id */
  int magic;        /**< Type of mailbox the email is in */
};


/* cb_format_str - Create the string to show in the sidebar */
static const char *cb_format_str(char *dest, size_t destlen, size_t col, int cols)
{
  return src;
}

