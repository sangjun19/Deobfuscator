	.file	"manuelzzz-gym_itp-IMD1012_questao_03_flatten.c"
	.text
	.globl	_TIG_IZ_C1Kz_argv
	.bss
	.align 8
	.type	_TIG_IZ_C1Kz_argv, @object
	.size	_TIG_IZ_C1Kz_argv, 8
_TIG_IZ_C1Kz_argv:
	.zero	8
	.globl	_TIG_IZ_C1Kz_envp
	.align 8
	.type	_TIG_IZ_C1Kz_envp, @object
	.size	_TIG_IZ_C1Kz_envp, 8
_TIG_IZ_C1Kz_envp:
	.zero	8
	.globl	_TIG_IZ_C1Kz_argc
	.align 4
	.type	_TIG_IZ_C1Kz_argc, @object
	.size	_TIG_IZ_C1Kz_argc, 4
_TIG_IZ_C1Kz_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d - %d - %d - %d - %d - %d\n"
.LC1:
	.string	" %[^\n]"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$208, %rsp
	movl	%edi, -180(%rbp)
	movq	%rsi, -192(%rbp)
	movq	%rdx, -200(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_C1Kz_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_C1Kz_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_C1Kz_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 92 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-C1Kz--0
# 0 "" 2
#NO_APP
	movl	-180(%rbp), %eax
	movl	%eax, _TIG_IZ_C1Kz_argc(%rip)
	movq	-192(%rbp), %rax
	movq	%rax, _TIG_IZ_C1Kz_argv(%rip)
	movq	-200(%rbp), %rax
	movq	%rax, _TIG_IZ_C1Kz_envp(%rip)
	nop
	movq	$1, -152(%rbp)
.L19:
	cmpq	$9, -152(%rbp)
	ja	.L22
	movq	-152(%rbp), %rax
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
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L22-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L22-.L8
	.long	.L22-.L8
	.long	.L7-.L8
	.text
.L13:
	movq	$2, -152(%rbp)
	jmp	.L15
.L11:
	movl	-168(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	tolower@PLT
	movl	%eax, -164(%rbp)
	movl	-164(%rbp), %eax
	movl	%eax, %edx
	movl	-168(%rbp), %eax
	cltq
	movb	%dl, -112(%rbp,%rax)
	movl	-168(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	movsbl	%al, %edx
	leaq	-144(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	contarNumeroVogais
	addl	$1, -168(%rbp)
	movq	$9, -152(%rbp)
	jmp	.L15
.L7:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -160(%rbp)
	movq	$0, -152(%rbp)
	jmp	.L15
.L9:
	movl	-128(%rbp), %esi
	movl	-132(%rbp), %r8d
	movl	-136(%rbp), %edi
	movl	-140(%rbp), %ecx
	movl	-144(%rbp), %edx
	movl	-124(%rbp), %eax
	subq	$8, %rsp
	pushq	%rsi
	movl	%r8d, %r9d
	movl	%edi, %r8d
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addq	$16, %rsp
	movq	$5, -152(%rbp)
	jmp	.L15
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L20
	jmp	.L21
.L14:
	movl	-168(%rbp), %eax
	cltq
	cmpq	%rax, -160(%rbp)
	jbe	.L17
	movq	$3, -152(%rbp)
	jmp	.L15
.L17:
	movq	$6, -152(%rbp)
	jmp	.L15
.L12:
	movl	$0, -144(%rbp)
	movl	$0, -140(%rbp)
	movl	$0, -136(%rbp)
	movl	$0, -132(%rbp)
	movl	$0, -128(%rbp)
	movl	$0, -124(%rbp)
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -168(%rbp)
	movq	$9, -152(%rbp)
	jmp	.L15
.L22:
	nop
.L15:
	jmp	.L19
.L21:
	call	__stack_chk_fail@PLT
.L20:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.globl	contarNumeroVogais
	.type	contarNumeroVogais, @function
contarNumeroVogais:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, %eax
	movb	%al, -28(%rbp)
	movq	$7, -8(%rbp)
.L49:
	cmpq	$14, -8(%rbp)
	ja	.L50
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L26(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L26(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L26:
	.long	.L50-.L26
	.long	.L36-.L26
	.long	.L50-.L26
	.long	.L35-.L26
	.long	.L34-.L26
	.long	.L33-.L26
	.long	.L32-.L26
	.long	.L31-.L26
	.long	.L30-.L26
	.long	.L29-.L26
	.long	.L50-.L26
	.long	.L28-.L26
	.long	.L51-.L26
	.long	.L50-.L26
	.long	.L25-.L26
	.text
.L34:
	movsbl	-28(%rbp), %eax
	subl	$97, %eax
	cmpl	$20, %eax
	ja	.L37
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L39(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L39(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L39:
	.long	.L43-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L42-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L41-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L40-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L37-.L39
	.long	.L38-.L39
	.text
.L38:
	movq	$1, -8(%rbp)
	jmp	.L44
.L40:
	movq	$5, -8(%rbp)
	jmp	.L44
.L41:
	movq	$14, -8(%rbp)
	jmp	.L44
.L42:
	movq	$3, -8(%rbp)
	jmp	.L44
.L43:
	movq	$8, -8(%rbp)
	jmp	.L44
.L37:
	movq	$9, -8(%rbp)
	nop
.L44:
	jmp	.L45
.L25:
	movl	$2, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L45
.L30:
	movl	$0, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L45
.L36:
	movl	$4, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L45
.L35:
	movl	$1, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L45
.L28:
	movq	-24(%rbp), %rax
	movl	-12(%rbp), %edx
	movslq	%edx, %rdx
	movl	(%rax,%rdx,4), %eax
	leal	1(%rax), %ecx
	movq	-24(%rbp), %rax
	movl	-12(%rbp), %edx
	movslq	%edx, %rdx
	movl	%ecx, (%rax,%rdx,4)
	movq	-24(%rbp), %rax
	movl	20(%rax), %eax
	leal	1(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, 20(%rax)
	movq	$12, -8(%rbp)
	jmp	.L45
.L29:
	movq	$6, -8(%rbp)
	jmp	.L45
.L32:
	cmpl	$-1, -12(%rbp)
	je	.L47
	movq	$11, -8(%rbp)
	jmp	.L45
.L47:
	movq	$12, -8(%rbp)
	jmp	.L45
.L33:
	movl	$3, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L45
.L31:
	movl	$-1, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L45
.L50:
	nop
.L45:
	jmp	.L49
.L51:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	contarNumeroVogais, .-contarNumeroVogais
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
