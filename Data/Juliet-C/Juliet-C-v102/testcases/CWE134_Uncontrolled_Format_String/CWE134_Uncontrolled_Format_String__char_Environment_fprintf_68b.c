/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68b.c
Label Definition File: CWE134_Uncontrolled_Format_String.label.xml
Template File: sources-sinks-68b.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Copy a fixed string into data
 * Sinks: fprintf
 *    GoodSink: fprintf with "%s" as the second argument and data as the third
 *    BadSink : fprintf with data as the second argument
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#define ENV_VARIABLE "ADD"

#ifdef _WIN32
# define GETENV getenv
#else
# define GETENV getenv
#endif

extern char * CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_bad_data;
extern char * CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_goodG2B_data;
extern char * CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_goodB2G_data;

#ifndef OMITBAD

void CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68b_bad_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_bad_data;
    /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
    fprintf(stdout, data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68b_goodG2B_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_goodG2B_data;
    /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
    fprintf(stdout, data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68b_goodB2G_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_goodB2G_data;
    /* FIX: Specify the format disallowing a format string vulnerability */
    fprintf(stdout, "%s\n", data);
}

#endif /* OMITGOOD */
