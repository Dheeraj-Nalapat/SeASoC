/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_fromConsole_fprintf_34.c
Label Definition File: CWE134_Uncontrolled_Format_String.label.xml
Template File: sources-sinks-34.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: fromConsole Read input from the console
 * GoodSource: Copy a fixed string into data
 * Sinks: fprintf
 *    GoodSink: fprintf with "%s" as the second argument and data as the third
 *    BadSink : fprintf with data as the second argument
 * Flow Variant: 34 Data flow: use of a union containing two methods of accessing the same data (within the same function)
 *
 * */

#include "std_testcase.h"

typedef union
{
    char * a;
    char * b;
} CWE134_Uncontrolled_Format_String__char_fromConsole_fprintf_34_union_type;

#ifndef OMITBAD

void CWE134_Uncontrolled_Format_String__char_fromConsole_fprintf_34_bad()
{
    char * data;
    CWE134_Uncontrolled_Format_String__char_fromConsole_fprintf_34_union_type my_union;
    char data_buf[100] = "";
    data = data_buf;
    {
        /* Read input from the console */
        size_t data_len = strlen(data);
        /* if there is room in data, read into it from the console */
        if(100-data_len > 1)
        {
            fgets(data+data_len, (int)(100-data_len), stdin);
            /* The next 3 lines remove the carriage return from the string that is
             * inserted by fgets() */
            data_len = strlen(data);
            if (data_len > 0)
            {
                data[data_len-1] = '\0';
            }
        }
    }
    my_union.a = data;
    {
        char * data = my_union.b;
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        fprintf(stdout, data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    CWE134_Uncontrolled_Format_String__char_fromConsole_fprintf_34_union_type my_union;
    char data_buf[100] = "";
    data = data_buf;
    /* FIX: Use a fixed string that does not contain a format specifier */
    strcpy(data, "fixedstringtest");
    my_union.a = data;
    {
        char * data = my_union.b;
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        fprintf(stdout, data);
    }
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G()
{
    char * data;
    CWE134_Uncontrolled_Format_String__char_fromConsole_fprintf_34_union_type my_union;
    char data_buf[100] = "";
    data = data_buf;
    {
        /* Read input from the console */
        size_t data_len = strlen(data);
        /* if there is room in data, read into it from the console */
        if(100-data_len > 1)
        {
            fgets(data+data_len, (int)(100-data_len), stdin);
            /* The next 3 lines remove the carriage return from the string that is
             * inserted by fgets() */
            data_len = strlen(data);
            if (data_len > 0)
            {
                data[data_len-1] = '\0';
            }
        }
    }
    my_union.a = data;
    {
        char * data = my_union.b;
        /* FIX: Specify the format disallowing a format string vulnerability */
        fprintf(stdout, "%s\n", data);
    }
}

void CWE134_Uncontrolled_Format_String__char_fromConsole_fprintf_34_good()
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
    CWE134_Uncontrolled_Format_String__char_fromConsole_fprintf_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE134_Uncontrolled_Format_String__char_fromConsole_fprintf_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
