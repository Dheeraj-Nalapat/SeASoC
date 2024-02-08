/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_fromFile_vsnprintf_68b.c
Label Definition File: CWE134_Uncontrolled_Format_String.vasinks.label.xml
Template File: sources-vasinks-68b.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: fromFile Read input from a file
 * GoodSource: Copy a fixed string into data
 * Sinks: vsnprintf
 *    GoodSink: vsnprintf with a format string
 *    BadSink : vsnprintf without a format string
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include <stdarg.h>
#include "std_testcase.h"

#ifdef _WIN32
# define FOPEN fopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

extern char * CWE134_Uncontrolled_Format_String__char_fromFile_vsnprintf_68_bad_data;
extern char * CWE134_Uncontrolled_Format_String__char_fromFile_vsnprintf_68_goodG2B_data;
extern char * CWE134_Uncontrolled_Format_String__char_fromFile_vsnprintf_68_goodB2G_data;

#ifndef OMITBAD

static void bad_vasink(char * data, ...)
{
    {
        char dest[100] = "";
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vsnprintf(dest, 100-1, data, args);
        va_end(args);
        printLine(dest);
    }
}

void CWE134_Uncontrolled_Format_String__char_fromFile_vsnprintf_68b_bad_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_fromFile_vsnprintf_68_bad_data;
    bad_vasink(data, data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B_vasink(char * data, ...)
{
    {
        char dest[100] = "";
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vsnprintf(dest, 100-1, data, args);
        va_end(args);
        printLine(dest);
    }
}

void CWE134_Uncontrolled_Format_String__char_fromFile_vsnprintf_68b_goodG2B_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_fromFile_vsnprintf_68_goodG2B_data;
    goodG2B_vasink(data, data);
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G_vasink(char * data, ...)
{
    {
        char dest[100] = "";
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vsnprintf(dest, 100-1, "%s", args);
        va_end(args);
        printLine(dest);
    }
}

void CWE134_Uncontrolled_Format_String__char_fromFile_vsnprintf_68b_goodB2G_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_fromFile_vsnprintf_68_goodB2G_data;
    goodB2G_vasink(data, data);
}

#endif /* OMITGOOD */
