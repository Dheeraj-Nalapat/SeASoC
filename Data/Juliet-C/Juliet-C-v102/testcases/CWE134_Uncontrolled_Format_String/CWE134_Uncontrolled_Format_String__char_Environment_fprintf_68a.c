/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68a.c
Label Definition File: CWE134_Uncontrolled_Format_String.label.xml
Template File: sources-sinks-68a.tmpl.c
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

char * CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_bad_data;
char * CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_goodG2B_data;
char * CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_goodB2G_data;

#ifndef OMITBAD

/* bad function declaration */
void CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68b_bad_sink();

void CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_bad()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    {
        /* Read input from an environment variable */
        size_t data_len = strlen(data);
        char * environment = GETENV(ENV_VARIABLE);
        /* If there is data in the environment variable */
        if (environment != NULL)
        {
            strncat(data+data_len, environment, 100-data_len-1);
        }
    }
    CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_bad_data = data;
    CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68b_bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declarations */
void CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68b_goodG2B_sink();
void CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68b_goodB2G_sink();

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    /* FIX: Use a fixed string that does not contain a format specifier */
    strcpy(data, "fixedstringtest");
    CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_goodG2B_data = data;
    CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68b_goodG2B_sink();
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    {
        /* Read input from an environment variable */
        size_t data_len = strlen(data);
        char * environment = GETENV(ENV_VARIABLE);
        /* If there is data in the environment variable */
        if (environment != NULL)
        {
            strncat(data+data_len, environment, 100-data_len-1);
        }
    }
    CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_goodB2G_data = data;
    CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68b_goodB2G_sink();
}

void CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_good()
{
    goodG2B();
    goodB2G();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE134_Uncontrolled_Format_String__char_Environment_fprintf_68_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
