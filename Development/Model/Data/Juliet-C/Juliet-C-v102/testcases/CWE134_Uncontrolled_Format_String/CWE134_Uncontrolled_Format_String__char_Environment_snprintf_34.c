/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_Environment_snprintf_34.c
Label Definition File: CWE134_Uncontrolled_Format_String.label.xml
Template File: sources-sinks-34.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Copy a fixed string into data
 * Sinks: snprintf
 *    GoodSink: snprintf with "%s" as the third argument and data as the fourth
 *    BadSink : snprintf with data as the third argument
 * Flow Variant: 34 Data flow: use of a union containing two methods of accessing the same data (within the same function)
 *
 * */

#include "std_testcase.h"

#define ENV_VARIABLE "ADD"

#ifdef _WIN32
# define GETENV getenv
#else
# define GETENV getenv
#endif

#ifdef _WIN32
# define SNPRINTF _snprintf
#else /* NOT _WIN32 */
# define SNPRINTF snprintf
#endif

typedef union
{
    char * a;
    char * b;
} CWE134_Uncontrolled_Format_String__char_Environment_snprintf_34_union_type;

#ifndef OMITBAD

void CWE134_Uncontrolled_Format_String__char_Environment_snprintf_34_bad()
{
    char * data;
    CWE134_Uncontrolled_Format_String__char_Environment_snprintf_34_union_type my_union;
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
    my_union.a = data;
    {
        char * data = my_union.b;
        {
            char dest[100] = "";
            /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
            SNPRINTF(dest, 100-1, data);
            printLine(dest);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    CWE134_Uncontrolled_Format_String__char_Environment_snprintf_34_union_type my_union;
    char data_buf[100] = "";
    data = data_buf;
    /* FIX: Use a fixed string that does not contain a format specifier */
    strcpy(data, "fixedstringtest");
    my_union.a = data;
    {
        char * data = my_union.b;
        {
            char dest[100] = "";
            /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
            SNPRINTF(dest, 100-1, data);
            printLine(dest);
        }
    }
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G()
{
    char * data;
    CWE134_Uncontrolled_Format_String__char_Environment_snprintf_34_union_type my_union;
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
    my_union.a = data;
    {
        char * data = my_union.b;
        {
            char dest[100] = "";
            /* FIX: Specify the format disallowing a format string vulnerability */
            SNPRINTF(dest, 100-1, "%s", data);
            printLine(dest);
        }
    }
}

void CWE134_Uncontrolled_Format_String__char_Environment_snprintf_34_good()
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
    CWE134_Uncontrolled_Format_String__char_Environment_snprintf_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE134_Uncontrolled_Format_String__char_Environment_snprintf_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif