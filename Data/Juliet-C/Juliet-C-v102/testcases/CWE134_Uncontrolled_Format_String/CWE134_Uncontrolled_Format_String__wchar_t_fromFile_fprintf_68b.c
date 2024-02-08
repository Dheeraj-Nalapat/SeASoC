/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__wchar_t_fromFile_fprintf_68b.c
Label Definition File: CWE134_Uncontrolled_Format_String.label.xml
Template File: sources-sinks-68b.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: fromFile Read input from a file
 * GoodSource: Copy a fixed string into data
 * Sinks: fprintf
 *    GoodSink: fwprintf with "%s" as the second argument and data as the third
 *    BadSink : fwprintf with data as the second argument
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define FOPEN _wfopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

extern wchar_t * CWE134_Uncontrolled_Format_String__wchar_t_fromFile_fprintf_68_bad_data;
extern wchar_t * CWE134_Uncontrolled_Format_String__wchar_t_fromFile_fprintf_68_goodG2B_data;
extern wchar_t * CWE134_Uncontrolled_Format_String__wchar_t_fromFile_fprintf_68_goodB2G_data;

#ifndef OMITBAD

void CWE134_Uncontrolled_Format_String__wchar_t_fromFile_fprintf_68b_bad_sink()
{
    wchar_t * data = CWE134_Uncontrolled_Format_String__wchar_t_fromFile_fprintf_68_bad_data;
    /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
    fwprintf(stdout, data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE134_Uncontrolled_Format_String__wchar_t_fromFile_fprintf_68b_goodG2B_sink()
{
    wchar_t * data = CWE134_Uncontrolled_Format_String__wchar_t_fromFile_fprintf_68_goodG2B_data;
    /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
    fwprintf(stdout, data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE134_Uncontrolled_Format_String__wchar_t_fromFile_fprintf_68b_goodB2G_sink()
{
    wchar_t * data = CWE134_Uncontrolled_Format_String__wchar_t_fromFile_fprintf_68_goodB2G_data;
    /* FIX: Specify the format disallowing a format string vulnerability */
    fwprintf(stdout, L"%s\n", data);
}

#endif /* OMITGOOD */
