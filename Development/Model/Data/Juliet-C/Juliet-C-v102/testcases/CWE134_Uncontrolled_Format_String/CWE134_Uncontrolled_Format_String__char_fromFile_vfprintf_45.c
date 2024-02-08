/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45.c
Label Definition File: CWE134_Uncontrolled_Format_String.vasinks.label.xml
Template File: sources-vasinks-45.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: fromFile Read input from a file
 * GoodSource: Copy a fixed string into data
 * Sinks: vfprintf
 *    GoodSink: vfprintf with a format string
 *    BadSink : vfprintf without a format string
 * Flow Variant: 45 Data flow: data passed as a static global variable from one function to another in the same source file
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

#ifndef OMITBAD

static char * CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_bad_data;
static char * CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_goodG2B_data;
static char * CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_goodB2G_data;

static void bad_vasink(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vfprintf(stdout, data, args);
        va_end(args);
    }
}

static void bad_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_bad_data;
    bad_vasink(data, data);
}

void CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_bad()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    {
        /* Read input from a file */
        size_t data_len = strlen(data);
        FILE * pFile;
        /* if there is room in data, attempt to read the input from a file */
        if(100-data_len > 1)
        {
            pFile = FOPEN("C:\\temp\\file.txt", "r");
            if (pFile != NULL)
            {
                fgets(data+data_len, (int)(100-data_len), pFile);
                fclose(pFile);
            }
        }
    }
    CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_bad_data = data;
    bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B_vasink(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vfprintf(stdout, data, args);
        va_end(args);
    }
}

static void goodG2B_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_goodG2B_data;
    goodG2B_vasink(data, data);
}

static void goodG2B()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    /* FIX: Use a fixed string that does not contain a format specifier */
    strcpy(data, "fixedstringtest");
    CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_goodG2B_data = data;
    goodG2B_sink();
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G_vasink(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vfprintf(stdout, "%s", args);
        va_end(args);
    }
}

static void goodB2G_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_goodB2G_data;
    goodB2G_vasink(data, data);
}

static void goodB2G()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    {
        /* Read input from a file */
        size_t data_len = strlen(data);
        FILE * pFile;
        /* if there is room in data, attempt to read the input from a file */
        if(100-data_len > 1)
        {
            pFile = FOPEN("C:\\temp\\file.txt", "r");
            if (pFile != NULL)
            {
                fgets(data+data_len, (int)(100-data_len), pFile);
                fclose(pFile);
            }
        }
    }
    CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_goodB2G_data = data;
    goodB2G_sink();
}

void CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_good()
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
    CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE134_Uncontrolled_Format_String__char_fromFile_vfprintf_45_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
