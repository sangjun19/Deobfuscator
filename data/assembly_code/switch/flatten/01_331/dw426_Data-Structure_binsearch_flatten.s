	.file	"dw426_Data-Structure_binsearch_flatten.c"
	.text
	.globl	_TIG_IZ_aPVR_envp
	.bss
	.align 8
	.type	_TIG_IZ_aPVR_envp, @object
	.size	_TIG_IZ_aPVR_envp, 8
_TIG_IZ_aPVR_envp:
	.zero	8
	.globl	_TIG_IZ_aPVR_argc
	.align 4
	.type	_TIG_IZ_aPVR_argc, @object
	.size	_TIG_IZ_aPVR_argc, 4
_TIG_IZ_aPVR_argc:
	.zero	4
	.globl	_TIG_IZ_aPVR_argv
	.align 8
	.type	_TIG_IZ_aPVR_argv, @object
	.size	_TIG_IZ_aPVR_argv, 8
_TIG_IZ_aPVR_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"\354\210\253\354\236\220 %d\353\212\224 \354\235\270\353\215\261\354\212\244 %d\354\227\220 \354\236\210\354\212\265\353\213\210\353\213\244.\n"
	.align 8
.LC1:
	.string	"\354\210\253\354\236\220 %d\353\212\224 \353\260\260\354\227\264\354\227\220 \354\227\206\354\212\265\353\213\210\353\213\244.\n"
	.align 8
.LC2:
	.string	"\354\260\276\352\263\240 \354\213\266\354\235\200 \354\210\253\354\236\220\353\245\274 \354\236\205\353\240\245\355\225\230\354\204\270\354\232\224: "
.LC3:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_aPVR_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_aPVR_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_aPVR_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-aPVR--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_aPVR_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_aPVR_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_aPVR_envp(%rip)
	nop
	movq	$2, -56(%rbp)
.L18:
	cmpq	$5, -56(%rbp)
	ja	.L21
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	-72(%rbp), %eax
	movl	-68(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -56(%rbp)
	jmp	.L14
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L19
	jmp	.L20
.L10:
	movl	-72(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -56(%rbp)
	jmp	.L14
.L7:
	movl	$1, -48(%rbp)
	movl	$3, -44(%rbp)
	movl	$5, -40(%rbp)
	movl	$7, -36(%rbp)
	movl	$9, -32(%rbp)
	movl	$11, -28(%rbp)
	movl	$13, -24(%rbp)
	movl	$15, -20(%rbp)
	movl	$17, -16(%rbp)
	movl	$19, -12(%rbp)
	movl	$10, -64(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-64(%rbp), %eax
	leal	-1(%rax), %edx
	movl	-72(%rbp), %esi
	leaq	-48(%rbp), %rax
	movl	%edx, %ecx
	movl	$0, %edx
	movq	%rax, %rdi
	call	binsearch
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	movl	%eax, -68(%rbp)
	movq	$0, -56(%rbp)
	jmp	.L14
.L13:
	cmpl	$-1, -68(%rbp)
	je	.L16
	movq	$4, -56(%rbp)
	jmp	.L14
.L16:
	movq	$3, -56(%rbp)
	jmp	.L14
.L11:
	movq	$5, -56(%rbp)
	jmp	.L14
.L21:
	nop
.L14:
	jmp	.L18
.L20:
	call	__stack_chk_fail@PLT
.L19:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	compare
	.type	compare, @function
compare:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$4, -8(%rbp)
.L36:
	cmpq	$4, -8(%rbp)
	ja	.L37
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L29-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L26-.L25
	.long	.L24-.L25
	.text
.L24:
	movl	-20(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L30
	movq	$0, -8(%rbp)
	jmp	.L32
.L30:
	movq	$1, -8(%rbp)
	jmp	.L32
.L28:
	movl	-20(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jne	.L33
	movq	$3, -8(%rbp)
	jmp	.L32
.L33:
	movq	$2, -8(%rbp)
	jmp	.L32
.L26:
	movl	$0, %eax
	jmp	.L35
.L29:
	movl	$-1, %eax
	jmp	.L35
.L27:
	movl	$1, %eax
	jmp	.L35
.L37:
	nop
.L32:
	jmp	.L36
.L35:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	compare, .-compare
	.globl	binsearch
	.type	binsearch, @function
binsearch:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$40, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movl	%ecx, -36(%rbp)
	movq	$0, -8(%rbp)
.L59:
	cmpq	$10, -8(%rbp)
	ja	.L60
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L41(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L41(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L41:
	.long	.L49-.L41
	.long	.L48-.L41
	.long	.L47-.L41
	.long	.L46-.L41
	.long	.L45-.L41
	.long	.L60-.L41
	.long	.L44-.L41
	.long	.L60-.L41
	.long	.L43-.L41
	.long	.L42-.L41
	.long	.L40-.L41
	.text
.L45:
	cmpl	$1, -12(%rbp)
	je	.L50
	cmpl	$1, -12(%rbp)
	jg	.L51
	cmpl	$-1, -12(%rbp)
	je	.L52
	cmpl	$0, -12(%rbp)
	je	.L53
	jmp	.L51
.L50:
	movq	$9, -8(%rbp)
	jmp	.L54
.L53:
	movq	$8, -8(%rbp)
	jmp	.L54
.L52:
	movq	$6, -8(%rbp)
	jmp	.L54
.L51:
	movq	$10, -8(%rbp)
	nop
.L54:
	jmp	.L55
.L43:
	movl	-16(%rbp), %eax
	jmp	.L56
.L48:
	movl	-32(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jg	.L57
	movq	$2, -8(%rbp)
	jmp	.L55
.L57:
	movq	$3, -8(%rbp)
	jmp	.L55
.L46:
	movl	$-1, %eax
	jmp	.L56
.L42:
	movl	-16(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -36(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L55
.L44:
	movl	-16(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -32(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L55
.L40:
	movq	$1, -8(%rbp)
	jmp	.L55
.L49:
	movq	$1, -8(%rbp)
	jmp	.L55
.L47:
	movl	-32(%rbp), %edx
	movl	-36(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	-28(%rbp), %edx
	movl	%edx, %esi
	movl	%eax, %edi
	call	compare
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L55
.L60:
	nop
.L55:
	jmp	.L59
.L56:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	binsearch, .-binsearch
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
