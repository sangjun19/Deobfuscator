	.file	"ddsnowboard_JavaProjects_c_flatten.c"
	.text
	.globl	lopts
	.bss
	.align 8
	.type	lopts, @object
	.size	lopts, 8
lopts:
	.zero	8
	.globl	temp_dir
	.align 8
	.type	temp_dir, @object
	.size	temp_dir, 8
temp_dir:
	.zero	8
	.local	len
	.comm	len,4,4
	.local	buf
	.comm	buf,8,8
	.globl	show_disassembly
	.align 4
	.type	show_disassembly, @object
	.size	show_disassembly, 4
show_disassembly:
	.zero	4
	.globl	use_debugger
	.align 4
	.type	use_debugger, @object
	.size	use_debugger, 4
use_debugger:
	.zero	4
	.globl	use_plusplus
	.align 4
	.type	use_plusplus, @object
	.size	use_plusplus, 4
use_plusplus:
	.zero	4
	.globl	spec_size
	.align 4
	.type	spec_size, @object
	.size	spec_size, 4
spec_size:
	.zero	4
	.globl	_TIG_IZ_e2E3_envp
	.align 8
	.type	_TIG_IZ_e2E3_envp, @object
	.size	_TIG_IZ_e2E3_envp, 8
_TIG_IZ_e2E3_envp:
	.zero	8
	.globl	src_lines
	.align 8
	.type	src_lines, @object
	.size	src_lines, 8
src_lines:
	.zero	8
	.globl	use_main
	.align 4
	.type	use_main, @object
	.size	use_main, 4
use_main:
	.zero	4
	.globl	oneliner
	.align 4
	.type	oneliner, @object
	.size	oneliner, 4
oneliner:
	.zero	4
	.globl	src_fp
	.align 8
	.type	src_fp, @object
	.size	src_fp, 8
src_fp:
	.zero	8
	.globl	keep_files
	.align 4
	.type	keep_files, @object
	.size	keep_files, 4
keep_files:
	.zero	4
	.globl	gcc
	.align 8
	.type	gcc, @object
	.size	gcc, 8
gcc:
	.zero	8
	.globl	exec_file
	.align 8
	.type	exec_file, @object
	.size	exec_file, 8
exec_file:
	.zero	8
	.globl	store_dir
	.align 8
	.type	store_dir, @object
	.size	store_dir, 8
store_dir:
	.zero	8
	.globl	_TIG_IZ_e2E3_argv
	.align 8
	.type	_TIG_IZ_e2E3_argv, @object
	.size	_TIG_IZ_e2E3_argv, 8
_TIG_IZ_e2E3_argv:
	.zero	8
	.globl	root_dir
	.align 8
	.type	root_dir, @object
	.size	root_dir, 8
root_dir:
	.zero	8
	.globl	c_file
	.align 8
	.type	c_file, @object
	.size	c_file, 8
c_file:
	.zero	8
	.globl	spec
	.align 32
	.type	spec, @object
	.size	spec, 65536
spec:
	.zero	65536
	.globl	_TIG_IZ_e2E3_argc
	.align 4
	.type	_TIG_IZ_e2E3_argc, @object
	.size	_TIG_IZ_e2E3_argc, 4
_TIG_IZ_e2E3_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"%s:%s: %s cannot be used in file\n"
	.text
	.type	assert_cmdline, @function
assert_cmdline:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movl	%edx, -36(%rbp)
	movq	$2, -8(%rbp)
.L9:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L10
	cmpq	$0, -8(%rbp)
	je	.L11
	cmpq	$1, -8(%rbp)
	jne	.L10
	movq	-24(%rbp), %rcx
	movl	-36(%rbp), %edx
	movq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$0, -8(%rbp)
	jmp	.L5
.L2:
	cmpq	$0, -32(%rbp)
	je	.L7
	movq	$1, -8(%rbp)
	jmp	.L5
.L7:
	movq	$0, -8(%rbp)
	jmp	.L5
.L10:
	nop
.L5:
	jmp	.L9
.L11:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	assert_cmdline, .-assert_cmdline
	.section	.rodata
.LC1:
	.string	"out of memory\n"
	.text
	.type	str_concat, @function
str_concat:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$1, -32(%rbp)
.L25:
	cmpq	$5, -32(%rbp)
	ja	.L27
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L15(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L15(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L15:
	.long	.L20-.L15
	.long	.L19-.L15
	.long	.L18-.L15
	.long	.L17-.L15
	.long	.L16-.L15
	.long	.L14-.L15
	.text
.L16:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rax
	addq	%rdx, %rax
	leaq	1(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$0, -32(%rbp)
	jmp	.L21
.L19:
	movq	$4, -32(%rbp)
	jmp	.L21
.L17:
	movq	-48(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	$5, -32(%rbp)
	jmp	.L21
.L14:
	movq	-40(%rbp), %rax
	jmp	.L26
.L20:
	cmpq	$0, -40(%rbp)
	jne	.L23
	movq	$2, -32(%rbp)
	jmp	.L21
.L23:
	movq	$3, -32(%rbp)
	jmp	.L21
.L18:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$3, -32(%rbp)
	jmp	.L21
.L27:
	nop
.L21:
	jmp	.L25
.L26:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	str_concat, .-str_concat
	.section	.rodata
.LC2:
	.string	".."
.LC3:
	.string	"/"
.LC4:
	.string	"."
.LC5:
	.string	"/cache"
	.text
	.type	update_cache, @function
update_cache:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$272, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -192(%rbp)
.L82:
	cmpq	$37, -192(%rbp)
	ja	.L85
	movq	-192(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L31(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L31(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L31:
	.long	.L57-.L31
	.long	.L85-.L31
	.long	.L56-.L31
	.long	.L55-.L31
	.long	.L54-.L31
	.long	.L53-.L31
	.long	.L52-.L31
	.long	.L51-.L31
	.long	.L50-.L31
	.long	.L49-.L31
	.long	.L85-.L31
	.long	.L85-.L31
	.long	.L48-.L31
	.long	.L85-.L31
	.long	.L86-.L31
	.long	.L46-.L31
	.long	.L85-.L31
	.long	.L45-.L31
	.long	.L44-.L31
	.long	.L43-.L31
	.long	.L42-.L31
	.long	.L85-.L31
	.long	.L41-.L31
	.long	.L40-.L31
	.long	.L39-.L31
	.long	.L85-.L31
	.long	.L38-.L31
	.long	.L85-.L31
	.long	.L37-.L31
	.long	.L36-.L31
	.long	.L85-.L31
	.long	.L35-.L31
	.long	.L34-.L31
	.long	.L33-.L31
	.long	.L85-.L31
	.long	.L32-.L31
	.long	.L85-.L31
	.long	.L30-.L31
	.text
.L44:
	movl	-136(%rbp), %eax
	andl	$61440, %eax
	cmpl	$16384, %eax
	jne	.L58
	movq	$33, -192(%rbp)
	jmp	.L60
.L58:
	movq	$4, -192(%rbp)
	jmp	.L60
.L54:
	movq	-216(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$26, -192(%rbp)
	jmp	.L60
.L46:
	cmpq	$0, -224(%rbp)
	je	.L62
	movq	$29, -192(%rbp)
	jmp	.L60
.L62:
	movq	$4, -192(%rbp)
	jmp	.L60
.L35:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-232(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$14, -192(%rbp)
	jmp	.L60
.L48:
	movl	$0, -264(%rbp)
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$0, -224(%rbp)
	movq	-248(%rbp), %rax
	movq	%rax, %rdi
	call	rewinddir@PLT
	movq	$4, -192(%rbp)
	jmp	.L60
.L50:
	cmpl	$0, -252(%rbp)
	jne	.L64
	movq	$18, -192(%rbp)
	jmp	.L60
.L64:
	movq	$4, -192(%rbp)
	jmp	.L60
.L40:
	movq	-240(%rbp), %rax
	addq	$19, %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -256(%rbp)
	movq	$9, -192(%rbp)
	jmp	.L60
.L55:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-216(%rbp), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -224(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -208(%rbp)
	movq	$35, -192(%rbp)
	jmp	.L60
.L39:
	cmpq	$0, -248(%rbp)
	je	.L66
	movq	$26, -192(%rbp)
	jmp	.L60
.L66:
	movq	$31, -192(%rbp)
	jmp	.L60
.L38:
	movq	-248(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -240(%rbp)
	movq	$28, -192(%rbp)
	jmp	.L60
.L49:
	cmpl	$0, -256(%rbp)
	jne	.L68
	movq	$26, -192(%rbp)
	jmp	.L60
.L68:
	movq	$32, -192(%rbp)
	jmp	.L60
.L43:
	movq	-216(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -192(%rbp)
	jmp	.L60
.L34:
	movq	-232(%rbp), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -184(%rbp)
	movq	-184(%rbp), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, -176(%rbp)
	movq	-240(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-176(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, -216(%rbp)
	leaq	-160(%rbp), %rdx
	movq	-216(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -252(%rbp)
	movq	$8, -192(%rbp)
	jmp	.L60
.L45:
	movq	-72(%rbp), %rax
	cmpq	%rax, -208(%rbp)
	jle	.L70
	movq	$3, -192(%rbp)
	jmp	.L60
.L70:
	movq	$35, -192(%rbp)
	jmp	.L60
.L52:
	cmpl	$0, -260(%rbp)
	jne	.L72
	movq	$26, -192(%rbp)
	jmp	.L60
.L72:
	movq	$23, -192(%rbp)
	jmp	.L60
.L41:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-216(%rbp), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -224(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -208(%rbp)
	movq	$35, -192(%rbp)
	jmp	.L60
.L37:
	cmpq	$0, -240(%rbp)
	je	.L74
	movq	$37, -192(%rbp)
	jmp	.L60
.L74:
	movq	$5, -192(%rbp)
	jmp	.L60
.L53:
	movq	-248(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$31, -192(%rbp)
	jmp	.L60
.L33:
	cmpq	$0, -224(%rbp)
	jne	.L76
	movq	$22, -192(%rbp)
	jmp	.L60
.L76:
	movq	$17, -192(%rbp)
	jmp	.L60
.L30:
	movq	-240(%rbp), %rax
	addq	$19, %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -260(%rbp)
	movq	$6, -192(%rbp)
	jmp	.L60
.L57:
	movq	$0, -248(%rbp)
	movq	$0, -224(%rbp)
	movq	$0, -208(%rbp)
	movl	$0, -264(%rbp)
	movq	root_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -168(%rbp)
	movq	-168(%rbp), %rax
	leaq	.LC5(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, -232(%rbp)
	movq	-232(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -248(%rbp)
	movq	$24, -192(%rbp)
	jmp	.L60
.L51:
	cmpq	$0, -200(%rbp)
	je	.L78
	movq	$12, -192(%rbp)
	jmp	.L60
.L78:
	movq	$19, -192(%rbp)
	jmp	.L60
.L32:
	addl	$1, -264(%rbp)
	movq	$20, -192(%rbp)
	jmp	.L60
.L36:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	remove_dir
	movq	-248(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -200(%rbp)
	movq	$7, -192(%rbp)
	jmp	.L60
.L56:
	movq	$0, -192(%rbp)
	jmp	.L60
.L42:
	cmpl	$127, -264(%rbp)
	jle	.L80
	movq	$15, -192(%rbp)
	jmp	.L60
.L80:
	movq	$4, -192(%rbp)
	jmp	.L60
.L85:
	nop
.L60:
	jmp	.L82
.L86:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L84
	call	__stack_chk_fail@PLT
.L84:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	update_cache, .-update_cache
	.type	split_tokens, @function
split_tokens:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	$27, -24(%rbp)
.L124:
	cmpq	$28, -24(%rbp)
	ja	.L126
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L90(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L90(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L90:
	.long	.L107-.L90
	.long	.L106-.L90
	.long	.L126-.L90
	.long	.L105-.L90
	.long	.L104-.L90
	.long	.L126-.L90
	.long	.L126-.L90
	.long	.L103-.L90
	.long	.L126-.L90
	.long	.L126-.L90
	.long	.L126-.L90
	.long	.L126-.L90
	.long	.L102-.L90
	.long	.L101-.L90
	.long	.L126-.L90
	.long	.L100-.L90
	.long	.L126-.L90
	.long	.L99-.L90
	.long	.L98-.L90
	.long	.L126-.L90
	.long	.L97-.L90
	.long	.L96-.L90
	.long	.L95-.L90
	.long	.L94-.L90
	.long	.L126-.L90
	.long	.L93-.L90
	.long	.L92-.L90
	.long	.L91-.L90
	.long	.L89-.L90
	.text
.L98:
	call	__ctype_b_loc@PLT
	movq	%rax, -32(%rbp)
	movq	$21, -24(%rbp)
	jmp	.L108
.L93:
	cmpq	$0, -48(%rbp)
	jne	.L109
	movq	$7, -24(%rbp)
	jmp	.L108
.L109:
	movq	$13, -24(%rbp)
	jmp	.L108
.L104:
	movq	-56(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$1, -56(%rbp)
	movq	-8(%rbp), %rax
	movb	$0, (%rax)
	movq	$0, -24(%rbp)
	jmp	.L108
.L100:
	addq	$1, -56(%rbp)
	movq	$23, -24(%rbp)
	jmp	.L108
.L102:
	movq	-56(%rbp), %rdx
	movq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, -48(%rbp)
	addq	$1, -56(%rbp)
	movq	$23, -24(%rbp)
	jmp	.L108
.L106:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L111
	movq	$3, -24(%rbp)
	jmp	.L108
.L111:
	movq	$12, -24(%rbp)
	jmp	.L108
.L94:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L113
	movq	$18, -24(%rbp)
	jmp	.L108
.L113:
	movq	$28, -24(%rbp)
	jmp	.L108
.L105:
	movq	-48(%rbp), %rax
	jmp	.L125
.L96:
	movq	-32(%rbp), %rax
	movq	(%rax), %rdx
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L116
	movq	$28, -24(%rbp)
	jmp	.L108
.L116:
	movq	$15, -24(%rbp)
	jmp	.L108
.L92:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L118
	movq	$17, -24(%rbp)
	jmp	.L108
.L118:
	movq	$1, -24(%rbp)
	jmp	.L108
.L101:
	movq	-48(%rbp), %rax
	movq	$0, (%rax)
	movq	$0, -24(%rbp)
	jmp	.L108
.L99:
	addq	$1, -56(%rbp)
	movq	$0, -24(%rbp)
	jmp	.L108
.L91:
	movq	$20, -24(%rbp)
	jmp	.L108
.L95:
	call	__ctype_b_loc@PLT
	movq	%rax, -40(%rbp)
	movq	$26, -24(%rbp)
	jmp	.L108
.L89:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L120
	movq	$3, -24(%rbp)
	jmp	.L108
.L120:
	movq	$4, -24(%rbp)
	jmp	.L108
.L107:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L122
	movq	$22, -24(%rbp)
	jmp	.L108
.L122:
	movq	$1, -24(%rbp)
	jmp	.L108
.L103:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$13, -24(%rbp)
	jmp	.L108
.L97:
	movl	$8, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	$25, -24(%rbp)
	jmp	.L108
.L126:
	nop
.L108:
	jmp	.L124
.L125:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	split_tokens, .-split_tokens
	.type	get_line, @function
get_line:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$15, -16(%rbp)
.L155:
	cmpq	$15, -16(%rbp)
	ja	.L156
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L130(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L130(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L130:
	.long	.L142-.L130
	.long	.L141-.L130
	.long	.L140-.L130
	.long	.L139-.L130
	.long	.L138-.L130
	.long	.L156-.L130
	.long	.L137-.L130
	.long	.L136-.L130
	.long	.L135-.L130
	.long	.L134-.L130
	.long	.L156-.L130
	.long	.L133-.L130
	.long	.L132-.L130
	.long	.L131-.L130
	.long	.L156-.L130
	.long	.L129-.L130
	.text
.L138:
	movq	buf(%rip), %rax
	testq	%rax, %rax
	jne	.L143
	movq	$0, -16(%rbp)
	jmp	.L145
.L143:
	movq	$13, -16(%rbp)
	jmp	.L145
.L129:
	movl	$0, -28(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L145
.L132:
	movl	len(%rip), %eax
	addl	$4096, %eax
	movl	%eax, len(%rip)
	movl	len(%rip), %eax
	movslq	%eax, %rdx
	movq	buf(%rip), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, buf(%rip)
	movq	$4, -16(%rbp)
	jmp	.L145
.L135:
	cmpl	$-1, -24(%rbp)
	je	.L146
	movq	$11, -16(%rbp)
	jmp	.L145
.L146:
	movq	$7, -16(%rbp)
	jmp	.L145
.L141:
	cmpl	$10, -24(%rbp)
	jne	.L148
	movq	$7, -16(%rbp)
	jmp	.L145
.L148:
	movq	$9, -16(%rbp)
	jmp	.L145
.L139:
	movq	buf(%rip), %rdx
	movl	-28(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$6, -16(%rbp)
	jmp	.L145
.L133:
	movl	len(%rip), %eax
	cmpl	%eax, -28(%rbp)
	jne	.L150
	movq	$12, -16(%rbp)
	jmp	.L145
.L150:
	movq	$13, -16(%rbp)
	jmp	.L145
.L134:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -24(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L145
.L131:
	movl	-28(%rbp), %eax
	movl	%eax, -20(%rbp)
	addl	$1, -28(%rbp)
	movq	buf(%rip), %rdx
	movl	-20(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movl	-24(%rbp), %edx
	movb	%dl, (%rax)
	movq	$1, -16(%rbp)
	jmp	.L145
.L137:
	movq	buf(%rip), %rax
	jmp	.L152
.L142:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$13, -16(%rbp)
	jmp	.L145
.L136:
	cmpl	$0, -28(%rbp)
	jne	.L153
	movq	$2, -16(%rbp)
	jmp	.L145
.L153:
	movq	$3, -16(%rbp)
	jmp	.L145
.L140:
	movl	$0, %eax
	jmp	.L152
.L156:
	nop
.L145:
	jmp	.L155
.L152:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	get_line, .-get_line
	.section	.rodata
.LC6:
	.string	"/cache/%08lx"
	.text
	.type	build_store_dir, @function
build_store_dir:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -48(%rbp)
.L170:
	cmpq	$9, -48(%rbp)
	ja	.L173
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L160(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L160(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L160:
	.long	.L174-.L160
	.long	.L164-.L160
	.long	.L173-.L160
	.long	.L163-.L160
	.long	.L162-.L160
	.long	.L173-.L160
	.long	.L173-.L160
	.long	.L173-.L160
	.long	.L161-.L160
	.long	.L159-.L160
	.text
.L162:
	movl	-68(%rbp), %eax
	cltq
	leaq	spec(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movzbl	%al, %edx
	movq	-64(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	movabsq	$4223091239536077, %rdx
	movq	%rcx, %rax
	mulq	%rdx
	movq	%rcx, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rdx, %rax
	shrq	$15, %rax
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	imulq	$65521, %rax, %rdx
	movq	%rcx, %rax
	subq	%rdx, %rax
	movq	%rax, -64(%rbp)
	movq	-56(%rbp), %rdx
	movq	-64(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	movabsq	$4223091239536077, %rdx
	movq	%rcx, %rax
	mulq	%rdx
	movq	%rcx, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rdx, %rax
	shrq	$15, %rax
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	imulq	$65521, %rax, %rdx
	movq	%rcx, %rax
	subq	%rdx, %rax
	movq	%rax, -56(%rbp)
	addl	$1, -68(%rbp)
	movq	$3, -48(%rbp)
	jmp	.L166
.L161:
	movq	-56(%rbp), %rax
	salq	$16, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	leaq	-32(%rbp), %rax
	leaq	.LC6(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movq	root_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -40(%rbp)
	leaq	-32(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, store_dir(%rip)
	movq	$0, -48(%rbp)
	jmp	.L166
.L164:
	movq	$9, -48(%rbp)
	jmp	.L166
.L163:
	movl	spec_size(%rip), %eax
	cmpl	%eax, -68(%rbp)
	jge	.L167
	movq	$4, -48(%rbp)
	jmp	.L166
.L167:
	movq	$8, -48(%rbp)
	jmp	.L166
.L159:
	movq	$1, -64(%rbp)
	movq	$0, -56(%rbp)
	movl	$0, -68(%rbp)
	movq	$3, -48(%rbp)
	jmp	.L166
.L173:
	nop
.L166:
	jmp	.L170
.L174:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L172
	call	__stack_chk_fail@PLT
.L172:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	build_store_dir, .-build_store_dir
	.section	.rodata
	.align 8
.LC7:
	.string	"unexpected response from waitpid : %s\n"
.LC8:
	.string	"%s: %s : %s\n"
	.text
	.type	call_proc, @function
call_proc:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movq	%rdi, -104(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$13, -64(%rbp)
.L203:
	cmpq	$18, -64(%rbp)
	ja	.L207
	movq	-64(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L178(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L178(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L178:
	.long	.L190-.L178
	.long	.L189-.L178
	.long	.L207-.L178
	.long	.L207-.L178
	.long	.L188-.L178
	.long	.L187-.L178
	.long	.L186-.L178
	.long	.L185-.L178
	.long	.L184-.L178
	.long	.L183-.L178
	.long	.L207-.L178
	.long	.L182-.L178
	.long	.L207-.L178
	.long	.L181-.L178
	.long	.L207-.L178
	.long	.L180-.L178
	.long	.L207-.L178
	.long	.L179-.L178
	.long	.L177-.L178
	.text
.L177:
	movq	$0, -64(%rbp)
	jmp	.L191
.L188:
	call	__errno_location@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	strerror@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$6, -64(%rbp)
	jmp	.L191
.L180:
	cmpl	$-1, -80(%rbp)
	je	.L192
	cmpl	$0, -80(%rbp)
	je	.L193
	jmp	.L205
.L192:
	movq	$7, -64(%rbp)
	jmp	.L195
.L193:
	movq	$9, -64(%rbp)
	jmp	.L195
.L205:
	movq	$18, -64(%rbp)
	nop
.L195:
	jmp	.L191
.L184:
	call	__errno_location@PLT
	movq	%rax, -72(%rbp)
	movq	$11, -64(%rbp)
	jmp	.L191
.L189:
	movl	-84(%rbp), %eax
	sarl	$8, %eax
	movzbl	%al, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L204
	jmp	.L206
.L182:
	movq	-72(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$4, %eax
	jne	.L197
	movq	$0, -64(%rbp)
	jmp	.L191
.L197:
	movq	$4, -64(%rbp)
	jmp	.L191
.L183:
	movq	-104(%rbp), %rax
	movq	(%rax), %rax
	movq	-104(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execvp@PLT
	call	__errno_location@PLT
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	strerror@PLT
	movq	%rax, -48(%rbp)
	movq	-104(%rbp), %rax
	movq	(%rax), %rdx
	movq	-48(%rbp), %rcx
	movq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$0, -64(%rbp)
	jmp	.L191
.L181:
	call	fork@PLT
	movl	%eax, -80(%rbp)
	movq	$15, -64(%rbp)
	jmp	.L191
.L179:
	cmpl	$0, -76(%rbp)
	jns	.L199
	movq	$8, -64(%rbp)
	jmp	.L191
.L199:
	movq	$6, -64(%rbp)
	jmp	.L191
.L186:
	movl	-84(%rbp), %eax
	andl	$127, %eax
	testl	%eax, %eax
	je	.L201
	movq	$5, -64(%rbp)
	jmp	.L191
.L201:
	movq	$1, -64(%rbp)
	jmp	.L191
.L187:
	movl	$0, %edi
	movl	$0, %eax
	call	cmd_error
	movq	$1, -64(%rbp)
	jmp	.L191
.L190:
	leaq	-84(%rbp), %rcx
	movl	-80(%rbp), %eax
	movl	$0, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	waitpid@PLT
	movl	%eax, -76(%rbp)
	movq	$17, -64(%rbp)
	jmp	.L191
.L185:
	call	__errno_location@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	strerror@PLT
	movq	%rax, -32(%rbp)
	movq	-104(%rbp), %rax
	movq	(%rax), %rdx
	movq	-32(%rbp), %rcx
	movq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$0, -64(%rbp)
	jmp	.L191
.L207:
	nop
.L191:
	jmp	.L203
.L206:
	call	__stack_chk_fail@PLT
.L204:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	call_proc, .-call_proc
	.type	add_spec, @function
add_spec:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$6, -8(%rbp)
.L224:
	cmpq	$7, -8(%rbp)
	ja	.L225
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L211(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L211(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L211:
	.long	.L225-.L211
	.long	.L217-.L211
	.long	.L216-.L211
	.long	.L226-.L211
	.long	.L214-.L211
	.long	.L226-.L211
	.long	.L212-.L211
	.long	.L226-.L211
	.text
.L214:
	movl	$-1, spec_size(%rip)
	movq	$7, -8(%rbp)
	jmp	.L218
.L217:
	movl	-28(%rbp), %eax
	movslq	%eax, %rdx
	movl	spec_size(%rip), %eax
	cltq
	leaq	spec(%rip), %rcx
	addq	%rax, %rcx
	movq	-24(%rbp), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	memcpy@PLT
	movl	spec_size(%rip), %edx
	movl	-28(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, spec_size(%rip)
	movq	$5, -8(%rbp)
	jmp	.L218
.L212:
	movl	spec_size(%rip), %eax
	cmpl	$-1, %eax
	jne	.L220
	movq	$3, -8(%rbp)
	jmp	.L218
.L220:
	movq	$2, -8(%rbp)
	jmp	.L218
.L216:
	movl	spec_size(%rip), %edx
	movl	-28(%rbp), %eax
	addl	%edx, %eax
	cmpl	$65536, %eax
	jbe	.L222
	movq	$4, -8(%rbp)
	jmp	.L218
.L222:
	movq	$1, -8(%rbp)
	jmp	.L218
.L225:
	nop
.L218:
	jmp	.L224
.L226:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	add_spec, .-add_spec
	.section	.rodata
.LC9:
	.string	"/SPECS"
.LC10:
	.string	"rb"
	.text
	.type	check_specs, @function
check_specs:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-65536(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$80, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -65568(%rbp)
.L253:
	cmpq	$12, -65568(%rbp)
	ja	.L256
	movq	-65568(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L230(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L230(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L230:
	.long	.L242-.L230
	.long	.L241-.L230
	.long	.L240-.L230
	.long	.L239-.L230
	.long	.L238-.L230
	.long	.L237-.L230
	.long	.L236-.L230
	.long	.L235-.L230
	.long	.L234-.L230
	.long	.L233-.L230
	.long	.L232-.L230
	.long	.L231-.L230
	.long	.L229-.L230
	.text
.L238:
	movl	-65604(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L254
	jmp	.L255
.L229:
	cmpq	$0, -65584(%rbp)
	je	.L244
	movq	$7, -65568(%rbp)
	jmp	.L246
.L244:
	movq	$5, -65568(%rbp)
	jmp	.L246
.L234:
	movq	-65584(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$5, -65568(%rbp)
	jmp	.L246
.L241:
	cmpl	$0, -65596(%rbp)
	jne	.L247
	movq	$10, -65568(%rbp)
	jmp	.L246
.L247:
	movq	$8, -65568(%rbp)
	jmp	.L246
.L239:
	movq	$11, -65568(%rbp)
	jmp	.L246
.L231:
	movl	$0, -65604(%rbp)
	movq	store_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -65560(%rbp)
	movq	-65560(%rbp), %rax
	leaq	.LC9(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, -65592(%rbp)
	movq	-65592(%rbp), %rax
	leaq	.LC10(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -65584(%rbp)
	movq	$12, -65568(%rbp)
	jmp	.L246
.L233:
	cmpl	$-1, -65600(%rbp)
	jne	.L249
	movq	$6, -65568(%rbp)
	jmp	.L246
.L249:
	movq	$8, -65568(%rbp)
	jmp	.L246
.L236:
	movl	spec_size(%rip), %eax
	movslq	%eax, %rdx
	leaq	-65552(%rbp), %rax
	movq	%rax, %rsi
	leaq	spec(%rip), %rax
	movq	%rax, %rdi
	call	memcmp@PLT
	movl	%eax, -65596(%rbp)
	movq	$1, -65568(%rbp)
	jmp	.L246
.L237:
	movq	-65592(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$4, -65568(%rbp)
	jmp	.L246
.L232:
	movl	$1, -65604(%rbp)
	movq	$8, -65568(%rbp)
	jmp	.L246
.L242:
	movl	spec_size(%rip), %eax
	cltq
	cmpq	%rax, -65576(%rbp)
	jne	.L251
	movq	$2, -65568(%rbp)
	jmp	.L246
.L251:
	movq	$8, -65568(%rbp)
	jmp	.L246
.L235:
	movl	spec_size(%rip), %eax
	movslq	%eax, %rdx
	movq	-65584(%rbp), %rcx
	leaq	-65552(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -65576(%rbp)
	movq	$0, -65568(%rbp)
	jmp	.L246
.L240:
	movq	-65584(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -65600(%rbp)
	movq	$9, -65568(%rbp)
	jmp	.L246
.L256:
	nop
.L246:
	jmp	.L253
.L255:
	call	__stack_chk_fail@PLT
.L254:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	check_specs, .-check_specs
	.section	.rodata
.LC11:
	.string	"/tmp/%08x"
	.align 8
.LC12:
	.string	"failed to create temporary directory.\n"
	.text
	.type	make_temp_dir, @function
make_temp_dir:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -48(%rbp)
.L275:
	cmpq	$12, -48(%rbp)
	ja	.L278
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L260(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L260(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L260:
	.long	.L278-.L260
	.long	.L268-.L260
	.long	.L267-.L260
	.long	.L266-.L260
	.long	.L265-.L260
	.long	.L278-.L260
	.long	.L264-.L260
	.long	.L263-.L260
	.long	.L279-.L260
	.long	.L278-.L260
	.long	.L279-.L260
	.long	.L278-.L260
	.long	.L259-.L260
	.text
.L265:
	call	rand@PLT
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %edx
	leaq	-22(%rbp), %rax
	leaq	.LC11(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movq	root_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -32(%rbp)
	leaq	-22(%rbp), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, temp_dir(%rip)
	movq	temp_dir(%rip), %rax
	movl	$511, %esi
	movq	%rax, %rdi
	call	mkdir@PLT
	movl	%eax, -60(%rbp)
	movq	$1, -48(%rbp)
	jmp	.L269
.L259:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$8, -48(%rbp)
	jmp	.L269
.L268:
	cmpl	$0, -60(%rbp)
	jne	.L271
	movq	$10, -48(%rbp)
	jmp	.L269
.L271:
	movq	$7, -48(%rbp)
	jmp	.L269
.L266:
	cmpl	$999, -64(%rbp)
	jg	.L273
	movq	$4, -48(%rbp)
	jmp	.L269
.L273:
	movq	$12, -48(%rbp)
	jmp	.L269
.L264:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -40(%rbp)
	call	getpid@PLT
	movl	%eax, -56(%rbp)
	movq	-40(%rbp), %rax
	xorl	-56(%rbp), %eax
	movl	%eax, %edi
	call	srand@PLT
	movl	$0, -64(%rbp)
	movq	$3, -48(%rbp)
	jmp	.L269
.L263:
	movq	temp_dir(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	addl	$1, -64(%rbp)
	movq	$3, -48(%rbp)
	jmp	.L269
.L267:
	movq	$6, -48(%rbp)
	jmp	.L269
.L278:
	nop
.L269:
	jmp	.L275
.L279:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L277
	call	__stack_chk_fail@PLT
.L277:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	make_temp_dir, .-make_temp_dir
	.type	str_dup, @function
str_dup:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$3, -24(%rbp)
.L293:
	cmpq	$5, -24(%rbp)
	ja	.L295
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L283(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L283(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L283:
	.long	.L288-.L283
	.long	.L287-.L283
	.long	.L286-.L283
	.long	.L285-.L283
	.long	.L284-.L283
	.long	.L282-.L283
	.text
.L284:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$1, -24(%rbp)
	jmp	.L289
.L287:
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	$0, -24(%rbp)
	jmp	.L289
.L285:
	movq	$2, -24(%rbp)
	jmp	.L289
.L282:
	cmpq	$0, -32(%rbp)
	jne	.L290
	movq	$4, -24(%rbp)
	jmp	.L289
.L290:
	movq	$1, -24(%rbp)
	jmp	.L289
.L288:
	movq	-32(%rbp), %rax
	jmp	.L294
.L286:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	addq	$1, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$5, -24(%rbp)
	jmp	.L289
.L295:
	nop
.L289:
	jmp	.L293
.L294:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	str_dup, .-str_dup
	.section	.rodata
	.align 8
.LC13:
	.ascii	"C  (pronounced  large-C)  is  a psuedo interpreter of the C "
	.ascii	"programming\nlanguage.\n\nWithout the need of manual compila"
	.ascii	"tion, developers can  rapidly  create\nscripts  or write one"
	.ascii	"-liners using the C programming language that runs\nat nativ"
	.ascii	"e-code speed.\n\nUsage: C [options] [sourcefile] [arguments]"
	.ascii	"\n\nOptions:\n -c<gcc_option>    pass a compiler option to G"
	.ascii	"CC\n -d                use debugger\n -e <expression>   exec"
	.ascii	"utes the expression\n -i<include_file>  add an include file\n"
	.ascii	" -k                keep temporary files\n -l<gcc_option>    "
	.ascii	"pass a linker option to GCC\n -m                use main fun"
	.ascii	"ction\n -p                use C++ (i"
	.string	"mplies -m)\n -S                show disassembly\n -h, --help        displays this help message\n --version         displays version number\n\nExamples:\n % C -cWall -cO2 -e 'printf(\"hello world\\n\")'\n % C -p -e 'int main(int,char**) { cout << \"hello\" << endl; }'\n"
	.text
	.type	usage, @function
usage:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L300:
	cmpq	$0, -8(%rbp)
	je	.L297
	cmpq	$1, -8(%rbp)
	jne	.L301
	movq	stdout(%rip), %rax
	movq	%rax, %rcx
	movl	$876, %edx
	movl	$1, %esi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$0, %edi
	call	exit@PLT
.L297:
	movq	$1, -8(%rbp)
	jmp	.L299
.L301:
	nop
.L299:
	jmp	.L300
	.cfi_endproc
.LFE13:
	.size	usage, .-usage
	.section	.rodata
	.align 8
.LC14:
	.string	"failed to write file: %s : %s\n"
.LC15:
	.string	"wb"
	.text
	.type	save_specs, @function
save_specs:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	$3, -32(%rbp)
.L315:
	cmpq	$6, -32(%rbp)
	ja	.L316
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L305(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L305(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L305:
	.long	.L310-.L305
	.long	.L316-.L305
	.long	.L309-.L305
	.long	.L308-.L305
	.long	.L307-.L305
	.long	.L317-.L305
	.long	.L304-.L305
	.text
.L307:
	call	__errno_location@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	strerror@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$6, -32(%rbp)
	jmp	.L311
.L308:
	movq	$2, -32(%rbp)
	jmp	.L311
.L304:
	movl	spec_size(%rip), %eax
	cltq
	movq	-40(%rbp), %rdx
	movq	%rdx, %rcx
	movq	%rax, %rdx
	movl	$1, %esi
	leaq	spec(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -32(%rbp)
	jmp	.L311
.L310:
	cmpq	$0, -40(%rbp)
	jne	.L313
	movq	$4, -32(%rbp)
	jmp	.L311
.L313:
	movq	$6, -32(%rbp)
	jmp	.L311
.L309:
	movq	temp_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	leaq	.LC9(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	leaq	.LC15(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -40(%rbp)
	movq	$0, -32(%rbp)
	jmp	.L311
.L316:
	nop
.L311:
	jmp	.L315
.L317:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	save_specs, .-save_specs
	.section	.rodata
.LC16:
	.string	"/LARGE_C-%u"
.LC17:
	.string	"/tmp"
.LC18:
	.string	"%s owned by somebody else\n"
.LC19:
	.string	"TMPDIR"
.LC20:
	.string	"failed to stat: %s : %s\n"
	.text
	.globl	setup_dir
	.type	setup_dir, @function
setup_dir:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$256, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$9, -232(%rbp)
.L348:
	cmpq	$17, -232(%rbp)
	ja	.L351
	movq	-232(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L321(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L321(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L321:
	.long	.L335-.L321
	.long	.L334-.L321
	.long	.L333-.L321
	.long	.L332-.L321
	.long	.L331-.L321
	.long	.L330-.L321
	.long	.L351-.L321
	.long	.L329-.L321
	.long	.L328-.L321
	.long	.L327-.L321
	.long	.L351-.L321
	.long	.L326-.L321
	.long	.L325-.L321
	.long	.L352-.L321
	.long	.L323-.L321
	.long	.L322-.L321
	.long	.L351-.L321
	.long	.L320-.L321
	.text
.L331:
	movl	$63, %edi
	call	umask@PLT
	movl	%eax, -240(%rbp)
	movl	-240(%rbp), %eax
	movl	%eax, -252(%rbp)
	movl	-256(%rbp), %edx
	leaq	-32(%rbp), %rax
	leaq	.LC16(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -184(%rbp)
	leaq	-32(%rbp), %rdx
	movq	-184(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, root_dir(%rip)
	movq	root_dir(%rip), %rax
	movl	$448, %esi
	movq	%rax, %rdi
	call	mkdir@PLT
	movl	%eax, -244(%rbp)
	movq	$11, -232(%rbp)
	jmp	.L336
.L323:
	cmpl	$0, -248(%rbp)
	je	.L337
	movq	$17, -232(%rbp)
	jmp	.L336
.L337:
	movq	$8, -232(%rbp)
	jmp	.L336
.L322:
	movq	root_dir(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$7, -232(%rbp)
	jmp	.L336
.L325:
	cmpl	$-1, -252(%rbp)
	je	.L339
	movq	$0, -232(%rbp)
	jmp	.L336
.L339:
	movq	$13, -232(%rbp)
	jmp	.L336
.L328:
	movl	-148(%rbp), %eax
	cmpl	%eax, -256(%rbp)
	je	.L341
	movq	$15, -232(%rbp)
	jmp	.L336
.L341:
	movq	$7, -232(%rbp)
	jmp	.L336
.L334:
	movq	root_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, root_dir(%rip)
	movq	$7, -232(%rbp)
	jmp	.L336
.L332:
	call	geteuid@PLT
	movl	%eax, -236(%rbp)
	movl	-236(%rbp), %eax
	movl	%eax, -256(%rbp)
	movl	$-1, -252(%rbp)
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, root_dir(%rip)
	movq	$5, -232(%rbp)
	jmp	.L336
.L326:
	cmpl	$0, -244(%rbp)
	je	.L343
	movq	$2, -232(%rbp)
	jmp	.L336
.L343:
	movq	$7, -232(%rbp)
	jmp	.L336
.L327:
	movq	$3, -232(%rbp)
	jmp	.L336
.L320:
	call	__errno_location@PLT
	movq	%rax, -224(%rbp)
	movq	-224(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	strerror@PLT
	movq	%rax, -216(%rbp)
	movq	root_dir(%rip), %rax
	movq	-216(%rbp), %rdx
	movq	%rax, %rsi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$8, -232(%rbp)
	jmp	.L336
.L330:
	movq	root_dir(%rip), %rax
	testq	%rax, %rax
	je	.L346
	movq	$1, -232(%rbp)
	jmp	.L336
.L346:
	movq	$4, -232(%rbp)
	jmp	.L336
.L335:
	movl	-252(%rbp), %eax
	movl	%eax, %edi
	call	umask@PLT
	movq	$13, -232(%rbp)
	jmp	.L336
.L329:
	movq	root_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -208(%rbp)
	movq	-208(%rbp), %rax
	leaq	.LC5(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, -200(%rbp)
	movq	-200(%rbp), %rax
	movl	$511, %esi
	movq	%rax, %rdi
	call	mkdir@PLT
	movq	-200(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	root_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -192(%rbp)
	movq	-192(%rbp), %rax
	leaq	.LC17(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, -200(%rbp)
	movq	-200(%rbp), %rax
	movl	$511, %esi
	movq	%rax, %rdi
	call	mkdir@PLT
	movq	-200(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$12, -232(%rbp)
	jmp	.L336
.L333:
	movq	root_dir(%rip), %rax
	leaq	-176(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	lstat@PLT
	movl	%eax, -248(%rbp)
	movq	$14, -232(%rbp)
	jmp	.L336
.L351:
	nop
.L336:
	jmp	.L348
.L352:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L350
	call	__stack_chk_fail@PLT
.L350:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	setup_dir, .-setup_dir
	.section	.rodata
.LC21:
	.string	"-e"
	.align 8
.LC22:
	.string	"multiple -e options not permitted.\n"
.LC23:
	.string	"-i"
.LC24:
	.string	"--"
.LC25:
	.string	"--version"
.LC26:
	.string	"#include \""
.LC27:
	.string	"\"\n"
.LC28:
	.string	"-m"
.LC29:
	.string	"-p"
.LC30:
	.string	"-k"
	.align 8
.LC31:
	.string	"%s not followed by a GCC argument\n"
.LC32:
	.string	"-S"
.LC33:
	.string	"-d"
.LC34:
	.string	"invalid option: %s\n"
	.align 8
.LC35:
	.string	"-e should be followed by an expression\n"
.LC36:
	.string	"-l"
.LC37:
	.string	"--help"
.LC38:
	.string	";\n"
.LC39:
	.string	"-h"
.LC40:
	.string	"-"
.LC41:
	.string	"-c"
.LC42:
	.string	"-g"
.LC43:
	.string	"unknown option: %s\n"
	.text
	.type	parse_args, @function
parse_args:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movq	%rdi, -136(%rbp)
	movq	%rsi, -144(%rbp)
	movl	%edx, -148(%rbp)
	movq	$62, -48(%rbp)
.L465:
	cmpq	$75, -48(%rbp)
	ja	.L467
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L356(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L356(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L356:
	.long	.L416-.L356
	.long	.L415-.L356
	.long	.L414-.L356
	.long	.L413-.L356
	.long	.L412-.L356
	.long	.L411-.L356
	.long	.L410-.L356
	.long	.L409-.L356
	.long	.L408-.L356
	.long	.L467-.L356
	.long	.L407-.L356
	.long	.L406-.L356
	.long	.L405-.L356
	.long	.L467-.L356
	.long	.L404-.L356
	.long	.L403-.L356
	.long	.L467-.L356
	.long	.L402-.L356
	.long	.L401-.L356
	.long	.L400-.L356
	.long	.L399-.L356
	.long	.L398-.L356
	.long	.L467-.L356
	.long	.L397-.L356
	.long	.L396-.L356
	.long	.L467-.L356
	.long	.L467-.L356
	.long	.L395-.L356
	.long	.L394-.L356
	.long	.L393-.L356
	.long	.L392-.L356
	.long	.L391-.L356
	.long	.L390-.L356
	.long	.L389-.L356
	.long	.L388-.L356
	.long	.L387-.L356
	.long	.L386-.L356
	.long	.L385-.L356
	.long	.L384-.L356
	.long	.L383-.L356
	.long	.L382-.L356
	.long	.L467-.L356
	.long	.L381-.L356
	.long	.L380-.L356
	.long	.L379-.L356
	.long	.L378-.L356
	.long	.L467-.L356
	.long	.L377-.L356
	.long	.L376-.L356
	.long	.L375-.L356
	.long	.L467-.L356
	.long	.L374-.L356
	.long	.L373-.L356
	.long	.L467-.L356
	.long	.L372-.L356
	.long	.L467-.L356
	.long	.L371-.L356
	.long	.L370-.L356
	.long	.L369-.L356
	.long	.L368-.L356
	.long	.L367-.L356
	.long	.L467-.L356
	.long	.L366-.L356
	.long	.L467-.L356
	.long	.L365-.L356
	.long	.L364-.L356
	.long	.L467-.L356
	.long	.L467-.L356
	.long	.L363-.L356
	.long	.L362-.L356
	.long	.L361-.L356
	.long	.L360-.L356
	.long	.L359-.L356
	.long	.L358-.L356
	.long	.L357-.L356
	.long	.L355-.L356
	.text
.L401:
	cmpl	$0, -104(%rbp)
	jne	.L417
	movq	$2, -48(%rbp)
	jmp	.L419
.L417:
	movq	$8, -48(%rbp)
	jmp	.L419
.L375:
	cmpl	$0, -116(%rbp)
	jne	.L420
	movq	$39, -48(%rbp)
	jmp	.L419
.L420:
	movq	$64, -48(%rbp)
	jmp	.L419
.L373:
	movq	-72(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$99, %al
	jne	.L422
	movq	$6, -48(%rbp)
	jmp	.L419
.L422:
	movq	$17, -48(%rbp)
	jmp	.L419
.L412:
	call	show_version
	movq	$0, -48(%rbp)
	jmp	.L419
.L392:
	movq	-72(%rbp), %rax
	leaq	.LC21(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -96(%rbp)
	movq	$68, -48(%rbp)
	jmp	.L419
.L366:
	movq	$0, -48(%rbp)
	jmp	.L419
.L404:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$47, -48(%rbp)
	jmp	.L419
.L403:
	movq	-136(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L424
	movq	$75, -48(%rbp)
	jmp	.L419
.L424:
	movq	$65, -48(%rbp)
	jmp	.L419
.L371:
	movq	-72(%rbp), %rax
	movl	$2, %edx
	leaq	.LC23(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -100(%rbp)
	movq	$27, -48(%rbp)
	jmp	.L419
.L391:
	movq	-72(%rbp), %rax
	leaq	.LC24(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -80(%rbp)
	movq	$37, -48(%rbp)
	jmp	.L419
.L405:
	movq	-72(%rbp), %rax
	leaq	.LC25(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -128(%rbp)
	movq	$40, -48(%rbp)
	jmp	.L419
.L362:
	movl	-148(%rbp), %edx
	movq	-144(%rbp), %rcx
	movq	-72(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	assert_cmdline
	movq	src_lines(%rip), %rax
	leaq	.LC26(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, src_lines(%rip)
	movq	-72(%rbp), %rax
	leaq	2(%rax), %rdx
	movq	src_lines(%rip), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, src_lines(%rip)
	movq	src_lines(%rip), %rax
	leaq	.LC27(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, src_lines(%rip)
	movq	$0, -48(%rbp)
	jmp	.L419
.L408:
	movq	-72(%rbp), %rax
	leaq	.LC28(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -108(%rbp)
	movq	$57, -48(%rbp)
	jmp	.L419
.L378:
	movq	-136(%rbp), %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L426
	movq	$10, -48(%rbp)
	jmp	.L419
.L426:
	movq	$19, -48(%rbp)
	jmp	.L419
.L372:
	movl	-148(%rbp), %edx
	movq	-144(%rbp), %rcx
	movq	-72(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	assert_cmdline
	movq	$15, -48(%rbp)
	jmp	.L419
.L415:
	cmpl	$0, -120(%rbp)
	jne	.L428
	movq	$3, -48(%rbp)
	jmp	.L419
.L428:
	movq	$12, -48(%rbp)
	jmp	.L419
.L397:
	subq	$8, -136(%rbp)
	movq	$19, -48(%rbp)
	jmp	.L419
.L361:
	movq	-72(%rbp), %rax
	leaq	.LC29(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -112(%rbp)
	movq	$21, -48(%rbp)
	jmp	.L419
.L413:
	movl	-148(%rbp), %edx
	movq	-144(%rbp), %rcx
	movq	-72(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	assert_cmdline
	call	usage
	movq	$0, -48(%rbp)
	jmp	.L419
.L396:
	cmpl	$0, -124(%rbp)
	jne	.L430
	movq	$72, -48(%rbp)
	jmp	.L419
.L430:
	movq	$71, -48(%rbp)
	jmp	.L419
.L398:
	cmpl	$0, -112(%rbp)
	jne	.L432
	movq	$60, -48(%rbp)
	jmp	.L419
.L432:
	movq	$38, -48(%rbp)
	jmp	.L419
.L386:
	movq	-72(%rbp), %rax
	leaq	.LC30(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -104(%rbp)
	movq	$18, -48(%rbp)
	jmp	.L419
.L370:
	cmpl	$0, -108(%rbp)
	jne	.L434
	movq	$73, -48(%rbp)
	jmp	.L419
.L434:
	movq	$70, -48(%rbp)
	jmp	.L419
.L363:
	cmpl	$0, -96(%rbp)
	jne	.L436
	movq	$54, -48(%rbp)
	jmp	.L419
.L436:
	movq	$56, -48(%rbp)
	jmp	.L419
.L406:
	movq	-56(%rbp), %rax
	movq	%rax, -64(%rbp)
	movq	$43, -48(%rbp)
	jmp	.L419
.L374:
	movq	-72(%rbp), %rax
	addq	$1, %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movb	$45, (%rax)
	movq	-64(%rbp), %rax
	movq	(%rax), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	-64(%rbp), %rdx
	movq	%rax, (%rdx)
	movq	$0, -48(%rbp)
	jmp	.L419
.L400:
	cmpq	$0, -144(%rbp)
	je	.L438
	movq	$59, -48(%rbp)
	jmp	.L419
.L438:
	movq	$34, -48(%rbp)
	jmp	.L419
.L390:
	movq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$33, -48(%rbp)
	jmp	.L419
.L402:
	leaq	lopts(%rip), %rax
	movq	%rax, -56(%rbp)
	movq	$11, -48(%rbp)
	jmp	.L419
.L382:
	cmpl	$0, -128(%rbp)
	jne	.L440
	movq	$4, -48(%rbp)
	jmp	.L419
.L440:
	movq	$29, -48(%rbp)
	jmp	.L419
.L367:
	movl	$1, use_main(%rip)
	movl	$1, use_plusplus(%rip)
	movq	$0, -48(%rbp)
	jmp	.L419
.L368:
	movq	-136(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L442
	movq	$74, -48(%rbp)
	jmp	.L419
.L442:
	movq	$34, -48(%rbp)
	jmp	.L419
.L410:
	leaq	gcc(%rip), %rax
	movq	%rax, -56(%rbp)
	movq	$11, -48(%rbp)
	jmp	.L419
.L395:
	cmpl	$0, -100(%rbp)
	jne	.L444
	movq	$69, -48(%rbp)
	jmp	.L419
.L444:
	movq	$36, -48(%rbp)
	jmp	.L419
.L384:
	movq	-72(%rbp), %rax
	leaq	.LC32(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -116(%rbp)
	movq	$49, -48(%rbp)
	jmp	.L419
.L369:
	movq	-72(%rbp), %rax
	leaq	.LC33(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -92(%rbp)
	movq	$44, -48(%rbp)
	jmp	.L419
.L388:
	movq	-136(%rbp), %rax
	jmp	.L466
.L357:
	movq	-136(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$34, -48(%rbp)
	jmp	.L419
.L355:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$65, -48(%rbp)
	jmp	.L419
.L376:
	movq	-72(%rbp), %rax
	movl	$2, %edx
	leaq	.LC36(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -84(%rbp)
	movq	$20, -48(%rbp)
	jmp	.L419
.L360:
	movq	-72(%rbp), %rax
	leaq	.LC37(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -120(%rbp)
	movq	$1, -48(%rbp)
	jmp	.L419
.L394:
	movq	-136(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L447
	movq	$32, -48(%rbp)
	jmp	.L419
.L447:
	movq	$33, -48(%rbp)
	jmp	.L419
.L364:
	movl	oneliner(%rip), %eax
	testl	%eax, %eax
	je	.L449
	movq	$14, -48(%rbp)
	jmp	.L419
.L449:
	movq	$47, -48(%rbp)
	jmp	.L419
.L377:
	movq	-136(%rbp), %rax
	movq	%rax, -24(%rbp)
	addq	$8, -136(%rbp)
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movq	src_lines(%rip), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, src_lines(%rip)
	movq	src_lines(%rip), %rax
	leaq	.LC38(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, src_lines(%rip)
	movl	$1, oneliner(%rip)
	movq	$0, -48(%rbp)
	jmp	.L419
.L358:
	movl	$1, use_main(%rip)
	movq	$0, -48(%rbp)
	jmp	.L419
.L379:
	cmpl	$0, -92(%rbp)
	jne	.L451
	movq	$7, -48(%rbp)
	jmp	.L419
.L451:
	movq	$30, -48(%rbp)
	jmp	.L419
.L411:
	cmpl	$0, -88(%rbp)
	jne	.L453
	movq	$52, -48(%rbp)
	jmp	.L419
.L453:
	movq	$48, -48(%rbp)
	jmp	.L419
.L359:
	movl	-148(%rbp), %edx
	movq	-144(%rbp), %rcx
	movq	-72(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	assert_cmdline
	call	usage
	movq	$0, -48(%rbp)
	jmp	.L419
.L389:
	movq	-136(%rbp), %rax
	movq	%rax, -16(%rbp)
	addq	$8, -136(%rbp)
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movq	-64(%rbp), %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	-64(%rbp), %rdx
	movq	%rax, (%rdx)
	movq	$0, -48(%rbp)
	jmp	.L419
.L385:
	cmpl	$0, -80(%rbp)
	jne	.L455
	movq	$19, -48(%rbp)
	jmp	.L419
.L455:
	movq	$42, -48(%rbp)
	jmp	.L419
.L365:
	movq	-72(%rbp), %rax
	leaq	.LC39(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -124(%rbp)
	movq	$24, -48(%rbp)
	jmp	.L419
.L407:
	movq	-136(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$8, -136(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	leaq	.LC40(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -76(%rbp)
	movq	$35, -48(%rbp)
	jmp	.L419
.L381:
	movq	-72(%rbp), %rax
	movl	$2, %edx
	leaq	.LC41(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -88(%rbp)
	movq	$5, -48(%rbp)
	jmp	.L419
.L416:
	movq	-136(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L457
	movq	$45, -48(%rbp)
	jmp	.L419
.L457:
	movq	$19, -48(%rbp)
	jmp	.L419
.L383:
	movq	gcc(%rip), %rax
	leaq	.LC32(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movl	$1, show_disassembly(%rip)
	movq	$0, -48(%rbp)
	jmp	.L419
.L409:
	movl	-148(%rbp), %edx
	movq	-144(%rbp), %rcx
	movq	-72(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	assert_cmdline
	movq	gcc(%rip), %rax
	leaq	.LC42(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movl	$1, use_debugger(%rip)
	movq	$0, -48(%rbp)
	jmp	.L419
.L387:
	cmpl	$0, -76(%rbp)
	jne	.L459
	movq	$23, -48(%rbp)
	jmp	.L419
.L459:
	movq	$31, -48(%rbp)
	jmp	.L419
.L393:
	movq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$0, -48(%rbp)
	jmp	.L419
.L380:
	movq	-72(%rbp), %rax
	addq	$2, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L461
	movq	$28, -48(%rbp)
	jmp	.L419
.L461:
	movq	$51, -48(%rbp)
	jmp	.L419
.L414:
	movl	$1, keep_files(%rip)
	movq	$0, -48(%rbp)
	jmp	.L419
.L399:
	cmpl	$0, -84(%rbp)
	jne	.L463
	movq	$52, -48(%rbp)
	jmp	.L419
.L463:
	movq	$58, -48(%rbp)
	jmp	.L419
.L467:
	nop
.L419:
	jmp	.L465
.L466:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	parse_args, .-parse_args
	.section	.rodata
.LC44:
	.string	"gdb"
.LC45:
	.string	"cannot open file: %s : %s\n"
.LC46:
	.string	"#!"
.LC47:
	.string	"/a.out"
.LC48:
	.string	"g++"
.LC49:
	.string	"/source.c"
.LC50:
	.string	"wt"
.LC51:
	.string	"rt"
	.align 8
.LC52:
	.string	"-D__LARGE_C_PREFIX__=int main(int argc, char** argv) {"
	.align 8
.LC53:
	.string	"-D__LARGE_C_SUFFIX__=; return 0; }"
.LC54:
	.string	"__LARGE_C_SUFFIX__\n"
.LC55:
	.string	"could not spawn child process"
.LC56:
	.string	"stdin"
.LC57:
	.string	"-D__LARGE_C_PREFIX__="
.LC58:
	.string	"-D__LARGE_C_SUFFIX__="
.LC59:
	.string	"option"
.LC60:
	.string	"-o"
.LC61:
	.string	"# 1 \"%s\" 1\n"
.LC62:
	.string	"could not execute compiler"
.LC63:
	.string	"// "
	.align 8
.LC64:
	.string	"failed to create temporary file: %s : %s\n"
.LC65:
	.string	"gcc"
.LC66:
	.string	"-I."
	.align 8
.LC67:
	.string	"#define __LARGE_C__ 0x00000500\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n#include <stdio.h>\n#include <stdlib.h>\n#ifdef __cplusplus\n}\n#include <iostream>\nusing namespace std;\n#endif\n\n__LARGE_C_PREFIX__\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB18:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$432, %rsp
	movl	%edi, -404(%rbp)
	movq	%rsi, -416(%rbp)
	movq	%rdx, -424(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, len(%rip)
	nop
.L469:
	movq	$0, buf(%rip)
	nop
.L470:
	movl	$0, spec_size(%rip)
	nop
.L471:
	movl	$0, -392(%rbp)
	jmp	.L472
.L473:
	movl	-392(%rbp), %eax
	cltq
	leaq	spec(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -392(%rbp)
.L472:
	cmpl	$65535, -392(%rbp)
	jle	.L473
	nop
.L474:
	movq	$0, lopts(%rip)
	nop
.L475:
	movq	$0, gcc(%rip)
	nop
.L476:
	movl	$0, show_disassembly(%rip)
	nop
.L477:
	movl	$0, keep_files(%rip)
	nop
.L478:
	movl	$0, use_plusplus(%rip)
	nop
.L479:
	movl	$0, use_main(%rip)
	nop
.L480:
	movl	$0, use_debugger(%rip)
	nop
.L481:
	movl	$0, oneliner(%rip)
	nop
.L482:
	movq	$0, src_fp(%rip)
	nop
.L483:
	movq	$0, src_lines(%rip)
	nop
.L484:
	movq	$0, c_file(%rip)
	nop
.L485:
	movq	$0, exec_file(%rip)
	nop
.L486:
	movq	$0, temp_dir(%rip)
	nop
.L487:
	movq	$0, store_dir(%rip)
	nop
.L488:
	movq	$0, root_dir(%rip)
	nop
.L489:
	movq	$0, _TIG_IZ_e2E3_envp(%rip)
	nop
.L490:
	movq	$0, _TIG_IZ_e2E3_argv(%rip)
	nop
.L491:
	movl	$0, _TIG_IZ_e2E3_argc(%rip)
	nop
	nop
.L492:
.L493:
#APP
# 728 "ddsnowboard_JavaProjects_c.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-e2E3--0
# 0 "" 2
#NO_APP
	movl	-404(%rbp), %eax
	movl	%eax, _TIG_IZ_e2E3_argc(%rip)
	movq	-416(%rbp), %rax
	movq	%rax, _TIG_IZ_e2E3_argv(%rip)
	movq	-424(%rbp), %rax
	movq	%rax, _TIG_IZ_e2E3_envp(%rip)
	nop
	movq	$2, -288(%rbp)
.L638:
	cmpq	$108, -288(%rbp)
	ja	.L641
	movq	-288(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L496(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L496(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L496:
	.long	.L573-.L496
	.long	.L572-.L496
	.long	.L571-.L496
	.long	.L641-.L496
	.long	.L641-.L496
	.long	.L570-.L496
	.long	.L569-.L496
	.long	.L641-.L496
	.long	.L568-.L496
	.long	.L567-.L496
	.long	.L566-.L496
	.long	.L565-.L496
	.long	.L641-.L496
	.long	.L564-.L496
	.long	.L563-.L496
	.long	.L641-.L496
	.long	.L562-.L496
	.long	.L641-.L496
	.long	.L641-.L496
	.long	.L561-.L496
	.long	.L641-.L496
	.long	.L560-.L496
	.long	.L559-.L496
	.long	.L558-.L496
	.long	.L557-.L496
	.long	.L556-.L496
	.long	.L555-.L496
	.long	.L554-.L496
	.long	.L553-.L496
	.long	.L641-.L496
	.long	.L641-.L496
	.long	.L552-.L496
	.long	.L551-.L496
	.long	.L550-.L496
	.long	.L549-.L496
	.long	.L548-.L496
	.long	.L641-.L496
	.long	.L547-.L496
	.long	.L641-.L496
	.long	.L546-.L496
	.long	.L641-.L496
	.long	.L545-.L496
	.long	.L544-.L496
	.long	.L641-.L496
	.long	.L543-.L496
	.long	.L542-.L496
	.long	.L641-.L496
	.long	.L641-.L496
	.long	.L541-.L496
	.long	.L641-.L496
	.long	.L540-.L496
	.long	.L539-.L496
	.long	.L538-.L496
	.long	.L537-.L496
	.long	.L536-.L496
	.long	.L535-.L496
	.long	.L534-.L496
	.long	.L533-.L496
	.long	.L641-.L496
	.long	.L532-.L496
	.long	.L531-.L496
	.long	.L530-.L496
	.long	.L529-.L496
	.long	.L528-.L496
	.long	.L527-.L496
	.long	.L526-.L496
	.long	.L525-.L496
	.long	.L524-.L496
	.long	.L641-.L496
	.long	.L523-.L496
	.long	.L522-.L496
	.long	.L641-.L496
	.long	.L521-.L496
	.long	.L520-.L496
	.long	.L519-.L496
	.long	.L518-.L496
	.long	.L641-.L496
	.long	.L641-.L496
	.long	.L517-.L496
	.long	.L516-.L496
	.long	.L515-.L496
	.long	.L514-.L496
	.long	.L641-.L496
	.long	.L641-.L496
	.long	.L641-.L496
	.long	.L513-.L496
	.long	.L512-.L496
	.long	.L641-.L496
	.long	.L511-.L496
	.long	.L510-.L496
	.long	.L509-.L496
	.long	.L508-.L496
	.long	.L507-.L496
	.long	.L506-.L496
	.long	.L641-.L496
	.long	.L505-.L496
	.long	.L504-.L496
	.long	.L641-.L496
	.long	.L641-.L496
	.long	.L503-.L496
	.long	.L641-.L496
	.long	.L502-.L496
	.long	.L641-.L496
	.long	.L501-.L496
	.long	.L500-.L496
	.long	.L499-.L496
	.long	.L498-.L496
	.long	.L497-.L496
	.long	.L495-.L496
	.text
.L540:
	cmpq	$0, -344(%rbp)
	jne	.L574
	movq	$25, -288(%rbp)
	jmp	.L576
.L574:
	movq	$32, -288(%rbp)
	jmp	.L576
.L515:
	movq	-296(%rbp), %rax
	leaq	.LC44(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, -296(%rbp)
	movq	$81, -288(%rbp)
	jmp	.L576
.L495:
	movl	use_debugger(%rip), %eax
	testl	%eax, %eax
	je	.L577
	movq	$80, -288(%rbp)
	jmp	.L576
.L577:
	movq	$81, -288(%rbp)
	jmp	.L576
.L500:
	cmpl	$0, -388(%rbp)
	je	.L579
	movq	$105, -288(%rbp)
	jmp	.L576
.L579:
	movq	$39, -288(%rbp)
	jmp	.L576
.L556:
	call	__errno_location@PLT
	movq	%rax, -176(%rbp)
	movq	-176(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	strerror@PLT
	movq	%rax, -168(%rbp)
	movq	-168(%rbp), %rdx
	movq	-336(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC45(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$32, -288(%rbp)
	jmp	.L576
.L538:
	leaq	.LC40(%rip), %rax
	movq	%rax, -304(%rbp)
	movq	$60, -288(%rbp)
	jmp	.L576
.L529:
	movq	-328(%rbp), %rax
	movl	$2, %edx
	leaq	.LC46(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -360(%rbp)
	movq	$14, -288(%rbp)
	jmp	.L576
.L501:
	movq	exec_file(%rip), %rax
	movq	%rax, -304(%rbp)
	movq	$60, -288(%rbp)
	jmp	.L576
.L498:
	movq	$0, -280(%rbp)
	movq	store_dir(%rip), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	utimes@PLT
	movq	store_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -272(%rbp)
	movq	-272(%rbp), %rax
	leaq	.LC47(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, exec_file(%rip)
	movq	exec_file(%rip), %rdx
	movq	-280(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, -280(%rbp)
	movq	-416(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	-280(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_merge
	movq	%rax, -280(%rbp)
	movq	exec_file(%rip), %rax
	movq	-280(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execv@PLT
	movq	exec_file(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	store_dir(%rip), %rax
	movq	%rax, %rdi
	call	remove_dir
	movq	$101, -288(%rbp)
	jmp	.L576
.L563:
	cmpl	$0, -360(%rbp)
	jne	.L581
	movq	$9, -288(%rbp)
	jmp	.L576
.L581:
	movq	$42, -288(%rbp)
	jmp	.L576
.L510:
	movq	gcc(%rip), %rax
	leaq	.LC48(%rip), %rdx
	movq	%rdx, (%rax)
	movq	$27, -288(%rbp)
	jmp	.L576
.L534:
	movq	$0, -296(%rbp)
	movq	$108, -288(%rbp)
	jmp	.L576
.L516:
	cmpl	$1, -376(%rbp)
	jne	.L583
	movq	$62, -288(%rbp)
	jmp	.L576
.L583:
	movq	$42, -288(%rbp)
	jmp	.L576
.L552:
	cmpl	$1, -368(%rbp)
	jne	.L585
	movq	$64, -288(%rbp)
	jmp	.L576
.L585:
	movq	$41, -288(%rbp)
	jmp	.L576
.L502:
	call	make_temp_dir
	movq	temp_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -264(%rbp)
	movq	-264(%rbp), %rax
	leaq	.LC47(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, exec_file(%rip)
	movq	temp_dir(%rip), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -256(%rbp)
	movq	-256(%rbp), %rax
	leaq	.LC49(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, c_file(%rip)
	movq	c_file(%rip), %rax
	leaq	.LC50(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, src_fp(%rip)
	movq	$61, -288(%rbp)
	jmp	.L576
.L523:
	movq	-344(%rbp), %rax
	movq	%rax, %rdi
	call	get_line
	movq	%rax, -328(%rbp)
	movq	$53, -288(%rbp)
	jmp	.L576
.L568:
	movq	-416(%rbp), %rdx
	movq	-352(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L587
	movq	$34, -288(%rbp)
	jmp	.L576
.L587:
	movq	$57, -288(%rbp)
	jmp	.L576
.L504:
	movq	-416(%rbp), %rax
	movq	%rax, -248(%rbp)
	addq	$8, -416(%rbp)
	movq	-248(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -336(%rbp)
	movq	-336(%rbp), %rax
	leaq	.LC51(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -344(%rbp)
	movq	$50, -288(%rbp)
	jmp	.L576
.L542:
	movq	gcc(%rip), %rax
	leaq	.LC52(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movq	gcc(%rip), %rax
	leaq	.LC53(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movq	$85, -288(%rbp)
	jmp	.L576
.L536:
	movl	use_plusplus(%rip), %eax
	testl	%eax, %eax
	je	.L589
	movq	$89, -288(%rbp)
	jmp	.L576
.L589:
	movq	$27, -288(%rbp)
	jmp	.L576
.L517:
	movq	src_fp(%rip), %rax
	movq	%rax, %rcx
	movl	$19, %edx
	movl	$1, %esi
	leaq	.LC54(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	src_fp(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$0, src_fp(%rip)
	movq	$54, -288(%rbp)
	jmp	.L576
.L572:
	movl	$0, -368(%rbp)
	addl	$1, -376(%rbp)
	movq	$79, -288(%rbp)
	jmp	.L576
.L514:
	movq	exec_file(%rip), %rdx
	movq	-296(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, -296(%rbp)
	movq	-416(%rbp), %rdx
	movq	-296(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_merge
	movq	%rax, -296(%rbp)
	movq	-296(%rbp), %rax
	leaq	.LC55(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	call_proc
	movl	%eax, -388(%rbp)
	movq	$19, -288(%rbp)
	jmp	.L576
.L558:
	movq	stdin(%rip), %rax
	movq	%rax, -344(%rbp)
	leaq	.LC56(%rip), %rax
	movq	%rax, -336(%rbp)
	movq	$69, -288(%rbp)
	jmp	.L576
.L522:
	movq	gcc(%rip), %rax
	leaq	.LC57(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movq	gcc(%rip), %rax
	leaq	.LC58(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movq	$85, -288(%rbp)
	jmp	.L576
.L562:
	movq	-312(%rbp), %rax
	movq	(%rax), %rax
	leaq	.LC59(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -364(%rbp)
	movq	$6, -288(%rbp)
	jmp	.L576
.L557:
	movq	stdin(%rip), %rax
	movq	%rax, -344(%rbp)
	addq	$8, -416(%rbp)
	leaq	.LC56(%rip), %rax
	movq	%rax, -336(%rbp)
	movq	$69, -288(%rbp)
	jmp	.L576
.L560:
	movl	$0, -376(%rbp)
	movq	$48, -288(%rbp)
	jmp	.L576
.L533:
	movl	keep_files(%rip), %eax
	testl	%eax, %eax
	jne	.L591
	movq	$107, -288(%rbp)
	jmp	.L576
.L591:
	movq	$44, -288(%rbp)
	jmp	.L576
.L513:
	movq	gcc(%rip), %rax
	leaq	.LC60(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movq	$99, -288(%rbp)
	jmp	.L576
.L555:
	movq	-328(%rbp), %rax
	addq	$1, %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -192(%rbp)
	movq	-192(%rbp), %rax
	movq	%rax, -320(%rbp)
	movq	-320(%rbp), %rax
	movq	%rax, %rdi
	call	split_tokens
	movq	%rax, -184(%rbp)
	movq	-184(%rbp), %rax
	movq	%rax, -312(%rbp)
	movq	$5, -288(%rbp)
	jmp	.L576
.L499:
	call	cleanup
	movl	-388(%rbp), %eax
	movl	%eax, %edi
	call	exit@PLT
.L565:
	movq	-416(%rbp), %rax
	movq	(%rax), %rax
	leaq	-160(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -384(%rbp)
	movq	$35, -288(%rbp)
	jmp	.L576
.L567:
	movl	$1, -368(%rbp)
	movq	$31, -288(%rbp)
	jmp	.L576
.L564:
	movq	-416(%rbp), %rax
	movq	(%rax), %rax
	leaq	.LC40(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -372(%rbp)
	movq	$55, -288(%rbp)
	jmp	.L576
.L528:
	movq	src_fp(%rip), %rax
	movq	%rax, %rsi
	movl	$10, %edi
	call	fputc@PLT
	movq	$33, -288(%rbp)
	jmp	.L576
.L539:
	movl	oneliner(%rip), %eax
	testl	%eax, %eax
	je	.L593
	movq	$67, -288(%rbp)
	jmp	.L576
.L593:
	movq	$11, -288(%rbp)
	jmp	.L576
.L497:
	movl	oneliner(%rip), %eax
	testl	%eax, %eax
	je	.L595
	movq	$51, -288(%rbp)
	jmp	.L576
.L595:
	movq	$91, -288(%rbp)
	jmp	.L576
.L561:
	movq	store_dir(%rip), %rax
	testq	%rax, %rax
	jne	.L597
	movq	$66, -288(%rbp)
	jmp	.L576
.L597:
	movq	$72, -288(%rbp)
	jmp	.L576
.L551:
	movq	src_fp(%rip), %rax
	movq	-336(%rbp), %rdx
	leaq	.LC61(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$69, -288(%rbp)
	jmp	.L576
.L509:
	call	cleanup
	movl	$0, %edi
	call	exit@PLT
.L524:
	call	build_store_dir
	movq	$44, -288(%rbp)
	jmp	.L576
.L535:
	cmpl	$0, -372(%rbp)
	jne	.L599
	movq	$24, -288(%rbp)
	jmp	.L576
.L599:
	movq	$96, -288(%rbp)
	jmp	.L576
.L531:
	movq	gcc(%rip), %rax
	movq	-304(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movq	c_file(%rip), %rdx
	movq	gcc(%rip), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movq	lopts(%rip), %rdx
	movq	gcc(%rip), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_merge
	movq	%rax, gcc(%rip)
	movq	gcc(%rip), %rax
	leaq	.LC62(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	call_proc
	movl	%eax, -388(%rbp)
	movq	$104, -288(%rbp)
	jmp	.L576
.L532:
	movq	src_lines(%rip), %rax
	movq	%rax, -208(%rbp)
	movq	src_lines(%rip), %rax
	addq	$8, %rax
	movq	%rax, src_lines(%rip)
	movq	src_fp(%rip), %rdx
	movq	-208(%rbp), %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fputs@PLT
	movq	$37, -288(%rbp)
	jmp	.L576
.L569:
	cmpl	$0, -364(%rbp)
	jne	.L601
	movq	$92, -288(%rbp)
	jmp	.L576
.L601:
	movq	$0, -288(%rbp)
	jmp	.L576
.L554:
	movl	use_main(%rip), %eax
	testl	%eax, %eax
	je	.L603
	movq	$70, -288(%rbp)
	jmp	.L576
.L603:
	movq	$45, -288(%rbp)
	jmp	.L576
.L530:
	movq	src_fp(%rip), %rax
	testq	%rax, %rax
	jne	.L605
	movq	$93, -288(%rbp)
	jmp	.L576
.L605:
	movq	$37, -288(%rbp)
	jmp	.L576
.L549:
	movq	-416(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -200(%rbp)
	movq	-200(%rbp), %rax
	addl	$1, %eax
	movl	%eax, %edx
	movq	-416(%rbp), %rax
	movq	(%rax), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	add_spec
	addq	$8, -416(%rbp)
	movq	$8, -288(%rbp)
	jmp	.L576
.L519:
	movq	src_lines(%rip), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L607
	movq	$59, -288(%rbp)
	jmp	.L576
.L607:
	movq	$10, -288(%rbp)
	jmp	.L576
.L518:
	call	cleanup
	movq	$28, -288(%rbp)
	jmp	.L576
.L541:
	movq	-416(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L609
	movq	$23, -288(%rbp)
	jmp	.L576
.L609:
	movq	$13, -288(%rbp)
	jmp	.L576
.L559:
	call	check_specs
	movl	%eax, -380(%rbp)
	movq	$86, -288(%rbp)
	jmp	.L576
.L553:
	movl	-388(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L639
	jmp	.L640
.L537:
	cmpq	$0, -328(%rbp)
	je	.L612
	movq	$1, -288(%rbp)
	jmp	.L576
.L612:
	movq	$63, -288(%rbp)
	jmp	.L576
.L526:
	movq	-416(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -216(%rbp)
	movq	-216(%rbp), %rax
	addl	$1, %eax
	movl	%eax, %edx
	movq	-416(%rbp), %rax
	movq	(%rax), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	add_spec
	leaq	-160(%rbp), %rax
	addq	$48, %rax
	movl	$8, %esi
	movq	%rax, %rdi
	call	add_spec
	leaq	-160(%rbp), %rax
	addq	$88, %rax
	movl	$8, %esi
	movq	%rax, %rdi
	call	add_spec
	call	build_store_dir
	movq	$44, -288(%rbp)
	jmp	.L576
.L520:
	cmpl	$0, -356(%rbp)
	je	.L614
	movq	$75, -288(%rbp)
	jmp	.L576
.L614:
	movq	$28, -288(%rbp)
	jmp	.L576
.L543:
	movq	store_dir(%rip), %rax
	testq	%rax, %rax
	je	.L616
	movq	$22, -288(%rbp)
	jmp	.L576
.L616:
	movq	$101, -288(%rbp)
	jmp	.L576
.L570:
	movq	-312(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L618
	movq	$16, -288(%rbp)
	jmp	.L576
.L618:
	movq	$0, -288(%rbp)
	jmp	.L576
.L508:
	movq	-416(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L620
	movq	$51, -288(%rbp)
	jmp	.L576
.L620:
	movq	$44, -288(%rbp)
	jmp	.L576
.L521:
	call	save_specs
	call	update_cache
	movq	store_dir(%rip), %rdx
	movq	temp_dir(%rip), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	rename@PLT
	movl	%eax, -356(%rbp)
	movq	$73, -288(%rbp)
	jmp	.L576
.L503:
	movl	show_disassembly(%rip), %eax
	testl	%eax, %eax
	je	.L622
	movq	$52, -288(%rbp)
	jmp	.L576
.L622:
	movq	$103, -288(%rbp)
	jmp	.L576
.L550:
	movq	-344(%rbp), %rax
	movq	stdin(%rip), %rdx
	cmpq	%rdx, %rax
	je	.L624
	movq	$88, -288(%rbp)
	jmp	.L576
.L624:
	movq	$78, -288(%rbp)
	jmp	.L576
.L547:
	movq	src_lines(%rip), %rax
	testq	%rax, %rax
	je	.L626
	movq	$74, -288(%rbp)
	jmp	.L576
.L626:
	movq	$10, -288(%rbp)
	jmp	.L576
.L527:
	movq	src_fp(%rip), %rax
	movq	%rax, %rcx
	movl	$3, %edx
	movl	$1, %esi
	leaq	.LC63(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$41, -288(%rbp)
	jmp	.L576
.L506:
	call	__errno_location@PLT
	movq	%rax, -232(%rbp)
	movq	-232(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	strerror@PLT
	movq	%rax, -224(%rbp)
	movq	c_file(%rip), %rax
	movq	-224(%rbp), %rdx
	movq	%rax, %rsi
	leaq	.LC64(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$37, -288(%rbp)
	jmp	.L576
.L545:
	movq	src_fp(%rip), %rdx
	movq	-328(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fputs@PLT
	movq	$69, -288(%rbp)
	jmp	.L576
.L505:
	call	setup_dir
	movq	gcc(%rip), %rax
	leaq	.LC65(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movq	gcc(%rip), %rax
	leaq	.LC66(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, gcc(%rip)
	movq	src_lines(%rip), %rax
	leaq	.LC67(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, src_lines(%rip)
	addq	$8, -416(%rbp)
	movq	-416(%rbp), %rax
	movl	$0, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	parse_args
	movq	%rax, -240(%rbp)
	movq	-240(%rbp), %rax
	movq	%rax, -352(%rbp)
	movq	$8, -288(%rbp)
	jmp	.L576
.L507:
	movq	-312(%rbp), %rax
	leaq	8(%rax), %rcx
	movl	-376(%rbp), %edx
	movq	-336(%rbp), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	parse_args
	movl	$1, -368(%rbp)
	movq	$0, -288(%rbp)
	jmp	.L576
.L566:
	movl	oneliner(%rip), %eax
	testl	%eax, %eax
	jne	.L628
	movq	$21, -288(%rbp)
	jmp	.L576
.L628:
	movq	$78, -288(%rbp)
	jmp	.L576
.L544:
	movq	-328(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$35, %al
	jne	.L630
	movq	$26, -288(%rbp)
	jmp	.L576
.L630:
	movq	$31, -288(%rbp)
	jmp	.L576
.L573:
	movq	-320(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-312(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$31, -288(%rbp)
	jmp	.L576
.L546:
	movl	show_disassembly(%rip), %eax
	testl	%eax, %eax
	je	.L632
	movq	$90, -288(%rbp)
	jmp	.L576
.L632:
	movq	$56, -288(%rbp)
	jmp	.L576
.L525:
	call	cleanup
	movq	$28, -288(%rbp)
	jmp	.L576
.L511:
	movq	-344(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$78, -288(%rbp)
	jmp	.L576
.L548:
	cmpl	$0, -384(%rbp)
	jne	.L634
	movq	$65, -288(%rbp)
	jmp	.L576
.L634:
	movq	$44, -288(%rbp)
	jmp	.L576
.L512:
	cmpl	$0, -380(%rbp)
	je	.L636
	movq	$106, -288(%rbp)
	jmp	.L576
.L636:
	movq	$101, -288(%rbp)
	jmp	.L576
.L571:
	movq	$95, -288(%rbp)
	jmp	.L576
.L641:
	nop
.L576:
	jmp	.L638
.L640:
	call	__stack_chk_fail@PLT
.L639:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	main, .-main
	.type	sa_concat, @function
sa_concat:
.LFB20:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$7, -16(%rbp)
.L662:
	cmpq	$12, -16(%rbp)
	ja	.L664
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L645(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L645(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L645:
	.long	.L664-.L645
	.long	.L653-.L645
	.long	.L652-.L645
	.long	.L651-.L645
	.long	.L650-.L645
	.long	.L649-.L645
	.long	.L664-.L645
	.long	.L648-.L645
	.long	.L664-.L645
	.long	.L647-.L645
	.long	.L664-.L645
	.long	.L646-.L645
	.long	.L644-.L645
	.text
.L650:
	movl	-28(%rbp), %eax
	movl	%eax, -24(%rbp)
	addl	$1, -28(%rbp)
	movl	-24(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movq	%rax, (%rdx)
	movl	-28(%rbp), %eax
	movl	%eax, -20(%rbp)
	addl	$1, -28(%rbp)
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	$0, (%rax)
	movq	$11, -16(%rbp)
	jmp	.L654
.L644:
	addl	$1, -28(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L654
.L653:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	cmd_error
	movq	$4, -16(%rbp)
	jmp	.L654
.L651:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L655
	movq	$12, -16(%rbp)
	jmp	.L654
.L655:
	movq	$2, -16(%rbp)
	jmp	.L654
.L646:
	movq	-40(%rbp), %rax
	jmp	.L663
.L647:
	cmpq	$0, -40(%rbp)
	je	.L658
	movq	$3, -16(%rbp)
	jmp	.L654
.L658:
	movq	$2, -16(%rbp)
	jmp	.L654
.L649:
	cmpq	$0, -40(%rbp)
	jne	.L660
	movq	$1, -16(%rbp)
	jmp	.L654
.L660:
	movq	$4, -16(%rbp)
	jmp	.L654
.L648:
	movl	$0, -28(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L654
.L652:
	movl	-28(%rbp), %eax
	addl	$2, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L654
.L664:
	nop
.L654:
	jmp	.L662
.L663:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE20:
	.size	sa_concat, .-sa_concat
	.type	sa_merge, @function
sa_merge:
.LFB21:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$6, -16(%rbp)
.L676:
	cmpq	$6, -16(%rbp)
	je	.L666
	cmpq	$6, -16(%rbp)
	ja	.L678
	cmpq	$2, -16(%rbp)
	je	.L668
	cmpq	$2, -16(%rbp)
	ja	.L678
	cmpq	$0, -16(%rbp)
	je	.L669
	cmpq	$1, -16(%rbp)
	jne	.L678
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L670
	movq	$0, -16(%rbp)
	jmp	.L672
.L670:
	movq	$2, -16(%rbp)
	jmp	.L672
.L666:
	cmpq	$0, -32(%rbp)
	je	.L673
	movq	$1, -16(%rbp)
	jmp	.L672
.L673:
	movq	$2, -16(%rbp)
	jmp	.L672
.L669:
	movq	-32(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$8, -32(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	sa_concat
	movq	%rax, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L672
.L668:
	movq	-24(%rbp), %rax
	jmp	.L677
.L678:
	nop
.L672:
	jmp	.L676
.L677:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE21:
	.size	sa_merge, .-sa_merge
	.type	cleanup, @function
cleanup:
.LFB24:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L690:
	cmpq	$3, -8(%rbp)
	je	.L680
	cmpq	$3, -8(%rbp)
	ja	.L691
	cmpq	$2, -8(%rbp)
	je	.L692
	cmpq	$2, -8(%rbp)
	ja	.L691
	cmpq	$0, -8(%rbp)
	je	.L683
	cmpq	$1, -8(%rbp)
	jne	.L691
	movq	temp_dir(%rip), %rax
	movq	%rax, %rdi
	call	remove_dir
	movq	$2, -8(%rbp)
	jmp	.L684
.L680:
	movl	keep_files(%rip), %eax
	testl	%eax, %eax
	jne	.L685
	movq	$0, -8(%rbp)
	jmp	.L684
.L685:
	movq	$2, -8(%rbp)
	jmp	.L684
.L683:
	movq	temp_dir(%rip), %rax
	testq	%rax, %rax
	je	.L687
	movq	$1, -8(%rbp)
	jmp	.L684
.L687:
	movq	$2, -8(%rbp)
	jmp	.L684
.L691:
	nop
.L684:
	jmp	.L690
.L692:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE24:
	.size	cleanup, .-cleanup
	.type	cmd_error, @function
cmd_error:
.LFB25:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$240, %rsp
	movq	%rdi, -232(%rbp)
	movq	%rsi, -168(%rbp)
	movq	%rdx, -160(%rbp)
	movq	%rcx, -152(%rbp)
	movq	%r8, -144(%rbp)
	movq	%r9, -136(%rbp)
	testb	%al, %al
	je	.L694
	movaps	%xmm0, -128(%rbp)
	movaps	%xmm1, -112(%rbp)
	movaps	%xmm2, -96(%rbp)
	movaps	%xmm3, -80(%rbp)
	movaps	%xmm4, -64(%rbp)
	movaps	%xmm5, -48(%rbp)
	movaps	%xmm6, -32(%rbp)
	movaps	%xmm7, -16(%rbp)
.L694:
	movq	%fs:40, %rax
	movq	%rax, -184(%rbp)
	xorl	%eax, %eax
	movq	$5, -216(%rbp)
.L709:
	cmpq	$8, -216(%rbp)
	ja	.L711
	movq	-216(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L697(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L697(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L697:
	.long	.L703-.L697
	.long	.L702-.L697
	.long	.L711-.L697
	.long	.L711-.L697
	.long	.L701-.L697
	.long	.L700-.L697
	.long	.L699-.L697
	.long	.L698-.L697
	.long	.L696-.L697
	.text
.L701:
	movq	stderr(%rip), %rax
	leaq	-208(%rbp), %rdx
	movq	-232(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	vfprintf@PLT
	movq	$1, -216(%rbp)
	jmp	.L704
.L696:
	movq	src_fp(%rip), %rax
	testq	%rax, %rax
	je	.L705
	movq	$6, -216(%rbp)
	jmp	.L704
.L705:
	movq	$7, -216(%rbp)
	jmp	.L704
.L702:
	movq	$8, -216(%rbp)
	jmp	.L704
.L699:
	movq	src_fp(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$7, -216(%rbp)
	jmp	.L704
.L700:
	movl	$8, -208(%rbp)
	movl	$48, -204(%rbp)
	leaq	16(%rbp), %rax
	movq	%rax, -200(%rbp)
	leaq	-176(%rbp), %rax
	movq	%rax, -192(%rbp)
	movq	$0, -216(%rbp)
	jmp	.L704
.L703:
	cmpq	$0, -232(%rbp)
	je	.L707
	movq	$4, -216(%rbp)
	jmp	.L704
.L707:
	movq	$1, -216(%rbp)
	jmp	.L704
.L698:
	call	cleanup
	movl	$255, %edi
	call	exit@PLT
.L711:
	nop
.L704:
	jmp	.L709
	.cfi_endproc
.LFE25:
	.size	cmd_error, .-cmd_error
	.section	.rodata
	.align 8
.LC68:
	.ascii	"C 0.06\n\nC"
	.string	"opyright (C) 2006 Cybozu Labs, Inc.\nThis is free software; see the source for copying conditions.  There is NO\nwarranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\nWritten by Kazuho Oku (http://labs.cybozu.co.jp/blog/kazuhoatwork/)\n"
	.text
	.type	show_version, @function
show_version:
.LFB26:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L716:
	cmpq	$0, -8(%rbp)
	je	.L713
	cmpq	$1, -8(%rbp)
	jne	.L717
	movq	$0, -8(%rbp)
	jmp	.L715
.L713:
	movq	stdout(%rip), %rax
	movq	%rax, %rcx
	movl	$265, %edx
	movl	$1, %esi
	leaq	.LC68(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$0, %edi
	call	exit@PLT
.L717:
	nop
.L715:
	jmp	.L716
	.cfi_endproc
.LFE26:
	.size	show_version, .-show_version
	.type	remove_dir, @function
remove_dir:
.LFB27:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	$2, -32(%rbp)
.L743:
	cmpq	$15, -32(%rbp)
	ja	.L744
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L721(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L721(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L721:
	.long	.L732-.L721
	.long	.L731-.L721
	.long	.L730-.L721
	.long	.L729-.L721
	.long	.L745-.L721
	.long	.L744-.L721
	.long	.L727-.L721
	.long	.L744-.L721
	.long	.L726-.L721
	.long	.L725-.L721
	.long	.L744-.L721
	.long	.L724-.L721
	.long	.L745-.L721
	.long	.L744-.L721
	.long	.L722-.L721
	.long	.L720-.L721
	.text
.L722:
	cmpl	$0, -52(%rbp)
	je	.L734
	movq	$11, -32(%rbp)
	jmp	.L736
.L734:
	movq	$6, -32(%rbp)
	jmp	.L736
.L720:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	rmdir@PLT
	movq	$4, -32(%rbp)
	jmp	.L736
.L726:
	cmpl	$0, -56(%rbp)
	je	.L737
	movq	$9, -32(%rbp)
	jmp	.L736
.L737:
	movq	$6, -32(%rbp)
	jmp	.L736
.L731:
	cmpq	$0, -48(%rbp)
	jne	.L739
	movq	$12, -32(%rbp)
	jmp	.L736
.L739:
	movq	$6, -32(%rbp)
	jmp	.L736
.L729:
	movq	-40(%rbp), %rax
	addq	$19, %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -56(%rbp)
	movq	$8, -32(%rbp)
	jmp	.L736
.L724:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	str_dup
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, -16(%rbp)
	movq	-40(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	str_concat
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	unlink@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$6, -32(%rbp)
	jmp	.L736
.L725:
	movq	-40(%rbp), %rax
	addq	$19, %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -52(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L736
.L727:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -40(%rbp)
	movq	$0, -32(%rbp)
	jmp	.L736
.L732:
	cmpq	$0, -40(%rbp)
	je	.L741
	movq	$3, -32(%rbp)
	jmp	.L736
.L741:
	movq	$15, -32(%rbp)
	jmp	.L736
.L730:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -48(%rbp)
	movq	$1, -32(%rbp)
	jmp	.L736
.L744:
	nop
.L736:
	jmp	.L743
.L745:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE27:
	.size	remove_dir, .-remove_dir
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
