	.file	"SSwapnild31_C-programming_ex29_count_vowels_in_string-sing_do_while_flatten.c"
	.text
	.globl	_TIG_IZ_1thH_argc
	.bss
	.align 4
	.type	_TIG_IZ_1thH_argc, @object
	.size	_TIG_IZ_1thH_argc, 4
_TIG_IZ_1thH_argc:
	.zero	4
	.globl	_TIG_IZ_1thH_argv
	.align 8
	.type	_TIG_IZ_1thH_argv, @object
	.size	_TIG_IZ_1thH_argv, 8
_TIG_IZ_1thH_argv:
	.zero	8
	.globl	_TIG_IZ_1thH_envp
	.align 8
	.type	_TIG_IZ_1thH_envp, @object
	.size	_TIG_IZ_1thH_envp, 8
_TIG_IZ_1thH_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"vowels count : %d\n"
.LC1:
	.string	"Enter string : "
.LC2:
	.string	"%[^\n]"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_1thH_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_1thH_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_1thH_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 129 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-1thH--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_1thH_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_1thH_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_1thH_envp(%rip)
	nop
	movq	$15, -56(%rbp)
.L58:
	cmpq	$27, -56(%rbp)
	ja	.L61
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
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L61-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L61-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L62-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L17:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$105, %al
	jne	.L34
	movq	$4, -56(%rbp)
	jmp	.L36
.L34:
	movq	$27, -56(%rbp)
	jmp	.L36
.L10:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$79, %al
	jne	.L37
	movq	$2, -56(%rbp)
	jmp	.L36
.L37:
	movq	$12, -56(%rbp)
	jmp	.L36
.L29:
	addl	$1, -60(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L36
.L20:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$117, %al
	jne	.L39
	movq	$8, -56(%rbp)
	jmp	.L36
.L39:
	movq	$19, -56(%rbp)
	jmp	.L36
.L19:
	movq	$10, -56(%rbp)
	jmp	.L36
.L22:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$85, %al
	jne	.L41
	movq	$0, -56(%rbp)
	jmp	.L36
.L41:
	movq	$7, -56(%rbp)
	jmp	.L36
.L25:
	addl	$1, -60(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L36
.L32:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	testb	%al, %al
	je	.L43
	movq	$22, -56(%rbp)
	jmp	.L36
.L43:
	movq	$3, -56(%rbp)
	jmp	.L36
.L12:
	addl	$1, -60(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L36
.L30:
	movl	-60(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$21, -56(%rbp)
	jmp	.L36
.L11:
	addl	$1, -60(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L36
.L9:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$101, %al
	jne	.L46
	movq	$11, -56(%rbp)
	jmp	.L36
.L46:
	movq	$18, -56(%rbp)
	jmp	.L36
.L23:
	addl	$1, -60(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L36
.L21:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$73, %al
	jne	.L48
	movq	$24, -56(%rbp)
	jmp	.L36
.L48:
	movq	$25, -56(%rbp)
	jmp	.L36
.L16:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$65, %al
	jne	.L50
	movq	$20, -56(%rbp)
	jmp	.L36
.L50:
	movq	$6, -56(%rbp)
	jmp	.L36
.L18:
	addl	$1, -60(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L36
.L27:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$69, %al
	jne	.L52
	movq	$5, -56(%rbp)
	jmp	.L36
.L52:
	movq	$13, -56(%rbp)
	jmp	.L36
.L7:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$111, %al
	jne	.L54
	movq	$17, -56(%rbp)
	jmp	.L36
.L54:
	movq	$14, -56(%rbp)
	jmp	.L36
.L13:
	movl	-64(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$97, %al
	jne	.L56
	movq	$23, -56(%rbp)
	jmp	.L36
.L56:
	movq	$26, -56(%rbp)
	jmp	.L36
.L28:
	addl	$1, -60(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L36
.L24:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -60(%rbp)
	movl	$0, -64(%rbp)
	movq	$1, -56(%rbp)
	jmp	.L36
.L33:
	addl	$1, -60(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L36
.L26:
	addl	$1, -64(%rbp)
	movq	$1, -56(%rbp)
	jmp	.L36
.L31:
	addl	$1, -60(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L36
.L15:
	addl	$1, -60(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L36
.L61:
	nop
.L36:
	jmp	.L58
.L62:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L60
	call	__stack_chk_fail@PLT
.L60:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
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
