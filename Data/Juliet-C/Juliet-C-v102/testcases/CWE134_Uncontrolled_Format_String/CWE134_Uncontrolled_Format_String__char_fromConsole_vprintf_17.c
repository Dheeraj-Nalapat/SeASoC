/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_fromConsole_vprintf_17.c
Label Definition File: CWE134_Uncontrolled_Format_String.vasinks.label.xml
Template File: sources-vasinks-17.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: fromConsole Read input from the console
 * GoodSource: Copy a fixed string into data
 * Sinks: vprintf
 *    GoodSink: vprintf with a format string
 *    BadSink : vprintf without a format string
 * Flow Variant: 17 Control flow: for loops
 *
 * */

#include <stdarg.h>
#include "std_testcase.h"

#ifndef OMITBAD

static void bad_vasinkb(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vprintf(data, args);
        va_end(args);
    }
}

/* INCIDENTAL: CWE 561 Dead Code, the function below is never called */
static void bad_vasinkg(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vprintf("%s", args);
        va_end(args);
    }
}

void CWE134_Uncontrolled_Format_String__char_fromConsole_vprintf_17_bad()
{
    int h,i,j,k;
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    for(h = 0; h < 0; h++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a fixed string that does not contain a format specifier */
        strcpy(data, "fixedstringtest");
    }
    for(i = 0; i < 1; i++)
    {
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
    }
    for(j = 0; j < 1; j++)
    {
        bad_vasinkb(data, data);
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        bad_vasinkg(data, data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* INCIDENTAL: CWE 561 Dead Code, the function below is never called */
static void goodB2G_vasinkb(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vprintf(data, args);
        va_end(args);
    }
}

static void goodB2G_vasinkg(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vprintf("%s", args);
        va_end(args);
    }
}

/* goodB2G() - use badsource and goodsink by changing the conditions on the third and fourth while statements */
static void goodB2G()
{
    int h,i,j,k;
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    for(h = 0; h < 0; h++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a fixed string that does not contain a format specifier */
        strcpy(data, "fixedstringtest");
    }
    for(i = 0; i < 1; i++)
    {
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
    }
    for(j = 0; j < 0; j++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        goodB2G_vasinkb(data, data);
    }
    for(k = 0; k < 1; k++)
    {
        goodB2G_vasinkg(data, data);
    }
}

/* INCIDENTAL: CWE 561 Dead Code, the function below is never called */
static void goodG2B_vasinkg(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vprintf("%s", args);
        va_end(args);
    }
}

static void goodG2B_vasinkb(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vprintf(data, args);
        va_end(args);
    }
}

/* goodG2B() - use goodsource and badsink by changing the conditions on the first and second while statements */
static void goodG2B()
{
    int h,i,j,k;
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    for(h = 0; h < 1; h++)
    {
        /* FIX: Use a fixed string that does not contain a format specifier */
        strcpy(data, "fixedstringtest");
    }
    for(i = 0; i < 0; i++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
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
    }
    for(j = 0; j < 1; j++)
    {
        goodG2B_vasinkb(data, data);
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        goodG2B_vasinkg(data, data);
    }
}

void CWE134_Uncontrolled_Format_String__char_fromConsole_vprintf_17_good()
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
    CWE134_Uncontrolled_Format_String__char_fromConsole_vprintf_17_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE134_Uncontrolled_Format_String__char_fromConsole_vprintf_17_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
