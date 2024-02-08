/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE761_Free_Pointer_Not_At_Start_Of_Buffer__wchar_t_Environment_52b.c
Label Definition File: CWE761_Free_Pointer_Not_At_Start_Of_Buffer.label.xml
Template File: source-sinks-52b.tmpl.c
*/
/*
 * @description
 * CWE: 761 Free Pointer Not At Start of Buffer
 * BadSource: Environment Read input from an environment variable
 * Sinks:
 *    GoodSink: free() memory correctly at the start of the buffer
 *    BadSink : free() memory not at the start of the buffer
 * Flow Variant: 52 Data flow: data passed as an argument from one function to another to another in three different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define SEARCH_CHAR L'S'
#define BAD_SRC_FIXED L"Fixed String" /* MAINTENANCE NOTE: This string must contain the SEARCH_CHAR */

#define ENV_VARIABLE L"ADD"

#ifdef _WIN32
# define GETENV _wgetenv
#else
# define GETENV wgetenv
#endif

#ifndef OMITBAD

/* bad function declaration */
void CWE761_Free_Pointer_Not_At_Start_Of_Buffer__wchar_t_Environment_52c_bad_sink(wchar_t * data);

void CWE761_Free_Pointer_Not_At_Start_Of_Buffer__wchar_t_Environment_52b_bad_sink(wchar_t * data)
{
    CWE761_Free_Pointer_Not_At_Start_Of_Buffer__wchar_t_Environment_52c_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G uses the BadSource with the GoodSink */
void CWE761_Free_Pointer_Not_At_Start_Of_Buffer__wchar_t_Environment_52c_goodB2G_sink(wchar_t * data);

void CWE761_Free_Pointer_Not_At_Start_Of_Buffer__wchar_t_Environment_52b_goodB2G_sink(wchar_t * data)
{
    CWE761_Free_Pointer_Not_At_Start_Of_Buffer__wchar_t_Environment_52c_goodB2G_sink(data);
}

#endif /* OMITGOOD */